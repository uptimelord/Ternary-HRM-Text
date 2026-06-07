$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$Batch = 250
$MaxTokens = 20000
$HardFrac = 0.55
$MaxSpecTries = 50000

function Invoke-Generator {
    param(
        [string] $Style,
        [int] $Count,
        [int] $Seed,
        [string] $Out,
        [string] $Sigs,
        [string[]] $Exclude
    )

    $args = @(
        "python", "generate_word_problems.py",
        "--n", "$Count",
        "--batch", "$Batch",
        "--style", "$Style",
        "--seed", "$Seed",
        "--hard-frac", "$HardFrac",
        "--out", "$Out",
        "--emit-sigs", "$Sigs",
        "--max-spec-tries", "$MaxSpecTries",
        "--max-tokens", "$MaxTokens",
        "--sleep-seconds", "0.02"
    )
    foreach ($path in $Exclude) {
        $args += @("--exclude-sigs", $path)
    }

    & rtk @args
    if ($LASTEXITCODE -ne 0) {
        throw "generator failed for $Style with exit code $LASTEXITCODE"
    }
}

Invoke-Generator direct 40000 166 train100_direct_40k.jsonl train100_direct_40k.sigs @("heldout_all.sigs")
Invoke-Generator word 30000 167 train100_word_30k.jsonl train100_word_30k.sigs @("heldout_all.sigs", "train100_direct_40k.sigs")
Invoke-Generator trace 20000 168 train100_trace_20k.jsonl train100_trace_20k.sigs @("heldout_all.sigs", "train100_direct_40k.sigs", "train100_word_30k.sigs")
Invoke-Generator noisy 10000 169 train100_noisy_10k.jsonl train100_noisy_10k.sigs @("heldout_all.sigs", "train100_direct_40k.sigs", "train100_word_30k.sigs", "train100_trace_20k.sigs")

$combineAndValidate = @'
from __future__ import annotations
import collections
import importlib.util
import json
import sys
from pathlib import Path

base = Path(".")
files = [
    base / "train100_direct_40k.jsonl",
    base / "train100_word_30k.jsonl",
    base / "train100_trace_20k.jsonl",
    base / "train100_noisy_10k.jsonl",
]
sig_files = [path.with_suffix(".sigs") for path in files]

(base / "train100_all_100k.jsonl").write_text(
    "".join(path.read_text(encoding="utf-8") for path in files),
    encoding="utf-8",
)
(base / "train100_all_100k.sigs").write_text(
    "".join(path.read_text(encoding="utf-8") for path in sig_files),
    encoding="utf-8",
)

rows = [
    json.loads(line)
    for line in (base / "train100_all_100k.jsonl").open(encoding="utf-8")
    if line.strip()
]
sigs = [row["spec_signature"] for row in rows]
held = set((base / "heldout_all.sigs").read_text(encoding="utf-8").splitlines())
style_counts = dict(collections.Counter(row["style"] for row in rows))
kind_counts = dict(collections.Counter(row["kind"] for row in rows))
leaks = sorted(set(sigs) & held)

gen_path = base / "generate_word_problems.py"
spec_obj = importlib.util.spec_from_file_location("exp66_word_problem_generator_check", gen_path)
gen = importlib.util.module_from_spec(spec_obj)
sys.modules[spec_obj.name] = gen
spec_obj.loader.exec_module(gen)

def spec_from_signature(row):
    kind, nums, ops = row["spec_signature"].split("|")
    operands = tuple(int(x) for x in nums.split(","))
    return gen.Spec(kind, operands, tuple(ops), int(row["answer"]), tuple(row["steps"]))

bad = []
for row in rows:
    ok, reason = gen.verify_example(row["prompt"], spec_from_signature(row), row["style"])
    if not ok:
        bad.append((row["id"], reason, row["prompt"]))

summary = {
    "rows": len(rows),
    "styles": style_counts,
    "kinds": kind_counts,
    "unique_sigs": len(set(sigs)),
    "duplicate_sigs": len(sigs) - len(set(sigs)),
    "heldout_sigs": len(held),
    "leaks": len(leaks),
    "verify_bad": len(bad),
    "sample_direct": next(row["prompt"] for row in rows if row["style"] == "direct"),
    "sample_word": next(row["prompt"] for row in rows if row["style"] == "word"),
    "sample_trace": next(row["prompt"] for row in rows if row["style"] == "trace"),
    "sample_noisy": next(row["prompt"] for row in rows if row["style"] == "noisy"),
}
print(json.dumps(summary, indent=2))

assert summary["rows"] == 100000, summary
assert summary["duplicate_sigs"] == 0, summary
assert summary["leaks"] == 0, summary
assert summary["verify_bad"] == 0, summary
'@

$combineAndValidate | rtk python -
if ($LASTEXITCODE -ne 0) {
    throw "100k combine/validate failed with exit code $LASTEXITCODE"
}
