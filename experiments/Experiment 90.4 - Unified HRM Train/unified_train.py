import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import argparse
import time
from pathlib import Path
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.unified_hrm import UnifiedBitNetHRM
from training.sft_lib import DEFAULT_TOKENIZER

import importlib.util
spec = importlib.util.spec_from_file_location(
    "shared_reachability", 
    str(REPO_ROOT / "experiments" / "Experiment 90.3 - Shared Reachability Machine" / "shared_reachability.py")
)
exp90_3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp90_3)

def extract_symbols(row):
    dom = row.get("domain")
    sch = row.get("schema", {})
    if dom == "comparative_order":
        return list(sch.get("objects", []))
    if dom == "logic_rules":
        syms = set(sch.get("facts", [])) | {sch.get("query")}
        for r in sch.get("rules", []):
            syms |= {r["if"], r["then"]}
        return sorted(s for s in syms if s)
    return []

def prepare_lm_batch(rows, tokenizer, max_len, device):
    b = len(rows)
    input_ids = torch.zeros(b, max_len, dtype=torch.long, device=device)
    loss_mask = torch.zeros(b, max_len, dtype=torch.bool, device=device)
    target_ids = torch.zeros(b, max_len, dtype=torch.long, device=device)
    text_mask = torch.zeros(b, max_len, dtype=torch.bool, device=device)
    
    dummy_batch = []
    tok_lists = []
    
    for i, row in enumerate(rows):
        prompt = row["input_text"] if "_text" not in row else row["_text"]
        answer = row.get("answer_text")
        if not answer:
            if row["domain"] == "comparative_order":
                answer = ", ".join(row["solution"]["order"])
            else:
                answer = str(row["solution"]["answer"])
                
        full_text = f"{prompt} Answer: {answer}"
        prompt_toks = tokenizer.encode(f"{prompt} Answer: ", add_special_tokens=False).ids
        full_toks = tokenizer.encode(full_text, add_special_tokens=False).ids[:max_len]
        
        n = len(full_toks)
        input_ids[i, :n] = torch.tensor(full_toks, device=device)
        text_mask[i, :n] = True
        
        target_toks = full_toks[1:] + [0]
        target_ids[i, :n] = torch.tensor(target_toks, device=device)
        
        ans_start = max(0, len(prompt_toks) - 1)
        if ans_start < n - 1:
            loss_mask[i, ans_start:n-1] = True
            
        tok_lists.append(full_toks)
        dummy_batch.append({
            "symbols": extract_symbols(row) if "symbols" not in row else row["symbols"],
            "domain": row["domain"]
        })
        
    tag_ids = exp90_3.symbol_tags(dummy_batch, tok_lists, tokenizer, max_len, device)
    sym_mask, dom_ids = exp90_3.encode_symbols(dummy_batch, device)
    
    return {
        "input_ids": input_ids,
        "text_mask": text_mask,
        "tag_ids": tag_ids,
        "sym_mask": sym_mask,
        "dom_ids": dom_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
        "raw_rows": rows
    }

def load_dataset(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rows.append(json.loads(line))
    return rows

@torch.no_grad()
def evaluate(model, eval_rows, tokenizer, max_len, device, batch_size=32):
    model.eval()
    correct = 0
    total = 0
    
    for start in range(0, len(eval_rows), batch_size):
        batch_rows = eval_rows[start:start+batch_size]
        batch = prepare_lm_batch(batch_rows, tokenizer, max_len, device)
        
        logits = model(
            batch["input_ids"], 
            batch["text_mask"], 
            batch["tag_ids"], 
            batch["sym_mask"], 
            batch["dom_ids"]
        )
        
        preds = logits.argmax(dim=-1)
        
        for i in range(len(batch_rows)):
            mask = batch["loss_mask"][i]
            if mask.sum() == 0: continue
            
            p = preds[i][mask]
            t = batch["target_ids"][i][mask]
            
            if torch.equal(p, t):
                correct += 1
            total += 1
            
    return correct / max(1, total)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tokenizer.get_vocab_size()
    
    train_rows = load_dataset(args.train_data)
    eval_rows = load_dataset(args.eval_data)
    
    model = UnifiedBitNetHRM(vocab_size=vocab_size, d_model=args.width, logic_iters=args.logic_iters).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    import random
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Starting End-to-End Unified Training on {device}...")
    t0 = time.perf_counter()
    
    for step in range(1, args.steps + 1):
        model.train()
        batch_rows = [train_rows[rng.randrange(len(train_rows))] for _ in range(args.batch_size)]
        batch = prepare_lm_batch(batch_rows, tokenizer, args.max_len, device)
        
        logits = model(
            batch["input_ids"], 
            batch["text_mask"], 
            batch["tag_ids"], 
            batch["sym_mask"], 
            batch["dom_ids"]
        )
        
        loss = F.cross_entropy(
            logits.view(-1, vocab_size), 
            batch["target_ids"].view(-1), 
            reduction="none"
        )
        
        mask = batch["loss_mask"].view(-1)
        loss = (loss * mask.float()).sum() / max(1.0, mask.float().sum().item())
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Clip grads since we have 10 iters in recurrence
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if step % args.log_interval == 0 or step == args.steps:
            vram = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
            print(f"step={step}/{args.steps} loss={loss.item():.4f} vram={vram:.1f}MB elapsed_min={(time.perf_counter()-t0)/60:.1f}", flush=True)
            
    final_acc = evaluate(model, eval_rows, tokenizer, args.max_len, device, batch_size=args.batch_size)
    print(f"Final combined_strict@1 = {final_acc:.4f}")
    
    # Save report
    report = {
        "steps": args.steps,
        "loss": loss.item(),
        "combined_strict@1": final_acc,
        "params": sum(p.numel() for p in model.parameters())
    }
    
    out_dir = Path("artifacts/exp90_4_unified_train")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, default=REPO_ROOT / "datasets" / "multidomain_schema" / "v2" / "train.jsonl")
    parser.add_argument("--eval-data", type=Path, default=REPO_ROOT / "datasets" / "multidomain_schema" / "v2" / "heldout.jsonl")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--logic-iters", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()
    train(args)
