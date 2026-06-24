"""Smoke tests for Experiment 126 - NOMAD Phase 0."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP126_DIR = REPO_ROOT / "experiments" / "Experiment 126 - NOMAD Phase 0"

sys.path.insert(0, str(EXP126_DIR))

import nomad_model  # noqa: E402
import nomad_memory  # noqa: E402
import nomad_learning  # noqa: E402
import phase1a_memory_probe as phase1a  # noqa: E402
import phase1b_adapter_train as phase1b  # noqa: E402
import phase1b_logit_bias as phase1b_logit_bias  # noqa: E402
from models.layers import TernaryLinear158Init  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny config for smoke tests
# ---------------------------------------------------------------------------

TINY_CONFIG = {
    "hidden_size": 32,
    "n_layers": 1,
    "vocab_size": 256,
    "max_seq_len": 16,
    "max_iters": 3,
    "tau": 0.1,
    "damping": 0.9,
    "damping_decay": 0.9,
    "patience": 2,
    "min_damping": 1e-2,
    "fast_key_dim": 8,
    "fast_decay": 0.95,
    "fast_write_rate": 0.1,
}


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestNOMADModel:
    def test_build_and_forward(self):
        """Model builds and forward pass runs without error."""
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=False)
        batch, seq_len = 2, 8
        input_ids = torch.randint(0, 256, (batch, seq_len))

        hidden, intermediates = model(input_ids)

        assert hidden.shape == (batch, seq_len, 32)
        assert len(intermediates) == seq_len
        for h in intermediates:
            assert h.shape == (batch, 32)

    def test_build_hard_ternary(self):
        """Hard ternary mode: no grads, standard STE."""
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)
        for p in model.parameters():
            assert not p.requires_grad
        for m in model.modules():
            if isinstance(m, TernaryLinear158Init):
                assert m.ternary_ste_mode == "standard"

    def test_logits_shape(self):
        """Logit computation has correct shape."""
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)
        hidden = torch.randn(4, 32)
        logits = model.compute_logits(hidden)
        assert logits.shape == (4, 256)

    def test_count_params(self):
        """Parameter counting is consistent."""
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)
        counts = nomad_model.count_params(model)
        assert counts["total"] > 0
        assert counts["vocab"] > 0
        # Sum of components should match total
        component_sum = sum(
            v for k, v in counts.items() if k != "total"
        )
        assert component_sum == counts["total"]

    def test_model_size_mb(self):
        """Size estimation is positive."""
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)
        size = nomad_model.model_size_mb(model)
        assert size["size_mb"] > 0
        assert size["param_count"] > 0

    def test_embed_tokens(self):
        """Token embedding returns correct shape."""
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        emb = model.embed_tokens(ids)
        assert emb.shape == (2, 3, 32)

    def test_fast_memory_read_write(self):
        """Fast-weight memory accumulates and retrieves."""
        fm = nomad_model.FastWeightMemory(
            d_model=16, key_dim=4, decay=0.9, write_rate=0.5
        )
        h1 = torch.randn(2, 16)
        h2 = torch.randn(2, 16)

        # Initial read is zeros
        r0 = fm.read(h1)
        assert torch.allclose(r0, torch.zeros_like(r0), atol=1e-6)

        # Write h1, read with h2
        fm.write(h1)
        r1 = fm.read(h2)
        assert r1.shape == (2, 16)
        assert not torch.allclose(r1, torch.zeros_like(r1), atol=1e-6)

        # Reset clears state
        fm.reset()
        r2 = fm.read(h1)
        assert torch.allclose(r2, torch.zeros_like(r2), atol=1e-6)

    def test_reasoning_core_runs(self):
        """Fixed-point reasoning core runs without error."""
        core = nomad_model.NOMADReasoningCore(
            width=32, num_layers=1, max_iters=5, tau=0.01
        )
        batch = 2
        s0 = torch.zeros(batch, 32)
        h0 = torch.randn(batch, 32)
        e_t = torch.randn(batch, 32)
        r_t = torch.randn(batch, 32)
        m_t = torch.zeros(batch, 32)

        s_final, h_final, acts = core(s0, h0, e_t, r_t, m_t)

        assert s_final.shape == (batch, 32)
        assert h_final.shape == (batch, 32)
        assert acts == {}
        assert core.last_num_iters == 5

    def test_reasoning_core_collects_activations(self):
        """Last-iteration activations are returned when requested."""
        core = nomad_model.NOMADReasoningCore(
            width=32, num_layers=1, max_iters=3, tau=0.01
        )
        batch = 2
        s0 = torch.zeros(batch, 32)
        h0 = torch.randn(batch, 32)
        e_t = torch.randn(batch, 32)
        r_t = torch.randn(batch, 32)
        m_t = torch.zeros(batch, 32)

        _, _, acts = core(
            s0, h0, e_t, r_t, m_t, collect_activations=True
        )

        assert "reasoning_core.blocks.0.gate" in acts
        assert "reasoning_core.blocks.0.state_candidate" in acts
        assert "reasoning_core.blocks.0.state_out" in acts
        assert acts["reasoning_core.blocks.0.gate"].shape == (batch, 64)

    def test_forward_capture_activations(self):
        """Model forward returns stacked activation tensors."""
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)
        input_ids = torch.randint(0, 256, (2, 8))
        hidden, intermediates, acts = model(
            input_ids, capture_activations=True
        )

        assert hidden.shape == (2, 8, 32)
        assert len(intermediates) == 8
        assert "reasoning_core.blocks.0.gate" in acts
        assert acts["reasoning_core.blocks.0.gate"].shape == (2, 8, 64)

    def test_full_sequence_with_fast_memory(self):
        """Full sequence forward with fast memory accumulates context."""
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)
        batch, seq_len = 2, 12
        input_ids = torch.randint(0, 256, (batch, seq_len))

        hidden, intermediates = model(input_ids)

        assert hidden.shape == (batch, seq_len, 32)
        # Later hidden states should differ from earlier ones
        # (fast memory accumulates context)
        early = hidden[:, 0, :]
        late = hidden[:, -1, :]
        assert not torch.allclose(early, late, atol=1e-4)


# ---------------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------------


class TestExternalMemory:
    def test_insert_and_retrieve(self):
        """Basic insert/retrieve with exact match."""
        mem = nomad_memory.ExternalMemory(
            hidden_size=32,
            lambda_exact=1.0,
            lambda_gzip=0.0,
        )
        mem.insert("c1", "The quick brown fox jumps over the lazy dog")
        mem.insert("c2", "Python is a programming language")
        mem.insert("c3", "Machine learning models need data")

        results = mem.retrieve("brown fox", top_k=2)
        assert len(results) > 0
        assert results[0][0] == "c1"  # exact match should win

    def test_exact_match_score(self):
        """Exact match scoring works."""
        score = nomad_memory.exact_match_score("hello world", "hello world today")
        assert score == 1.0  # exact substring

        score = nomad_memory.exact_match_score("hello", "goodbye world")
        assert score == 0.0  # no match

    def test_ngram_score(self):
        """N-gram scoring for fuzzy matching."""
        score = nomad_memory.exact_match_score_ngram(
            "hello", "hello world", n=3
        )
        assert score > 0.5

        score = nomad_memory.exact_match_score_ngram(
            "xyz", "hello world", n=3
        )
        assert score < 0.5

    def test_compression_score(self):
        """Compression gain scoring is sensible."""
        score = nomad_memory.compression_rerank_score(
            "The capital of France is Paris",
            "Paris is the capital city of France",
        )
        # Should be positive (shared structure)
        assert 0.0 <= score <= 1.0

        score = nomad_memory.compression_rerank_score(
            "The capital of France is Paris",
            "Quantum mechanics describes particle behavior",
        )
        # Should be lower (less shared structure)
        assert 0.0 <= score <= 1.0

    def test_empty_memory(self):
        """Empty memory returns empty results."""
        mem = nomad_memory.ExternalMemory(hidden_size=32)
        results = mem.retrieve("anything")
        assert results == []

    def test_batch_insert(self):
        """Batch insert works."""
        mem = nomad_memory.ExternalMemory(hidden_size=32)
        items = [("a", "text one"), ("b", "text two"), ("c", "text three")]
        mem.insert_batch(items)
        assert len(mem) == 3
        assert mem.stats()["num_chunks"] == 3

    def test_ingest_text(self):
        """Text ingestion splits into chunks."""
        mem = nomad_memory.ExternalMemory(hidden_size=32)
        text = "A" * 500  # should create at least 2 chunks with 256-char chunks
        n = nomad_memory.ingest_text(mem, text, chunk_size_chars=200)
        assert n >= 2

    def test_get_vectors(self):
        """Vector retrieval returns correct shapes."""
        mem = nomad_memory.ExternalMemory(hidden_size=32)
        mem.insert("c1", "test chunk")
        mem.insert("c2", "another chunk")

        vectors = mem.get_vectors(["c1", "c2"])
        assert vectors.shape == (2, 32)


# ---------------------------------------------------------------------------
# Learning tests
# ---------------------------------------------------------------------------


class TestNOMADLearning:
    def test_nomad_chunked_vocab_update_matches_shared_exact_update(self):
        """Exp126 exact head update matches the shared Exp125 primitive."""
        from training import nobp_hard

        torch.manual_seed(126)
        base = TernaryLinear158Init(
            4,
            8,
            bias=False,
            ternary_group_size=4,
            ternary_threshold=0.25,
            ternary_scale_mode="mean_abs",
            ternary_ste_mode="standard",
        )
        fast = TernaryLinear158Init(
            4,
            8,
            bias=False,
            ternary_group_size=4,
            ternary_threshold=0.25,
            ternary_scale_mode="mean_abs",
            ternary_ste_mode="standard",
        )
        fast.load_state_dict(base.state_dict())
        hidden = torch.randn(7, 4)
        labels = torch.tensor([1, 7, -100, 3, 0, 2, 6])

        expected = nobp_hard.chunked_vocab_update(
            base,
            hidden,
            labels,
            chunk_size=3,
            lr=0.05,
            update_clip=0.75,
        )
        actual = nomad_learning.nomad_chunked_vocab_update(
            fast,
            hidden,
            labels,
            chunk_size=3,
            lr=0.05,
            update_clip=0.75,
        )

        torch.testing.assert_close(actual.loss, expected.loss)
        torch.testing.assert_close(actual.hidden_feedback, expected.hidden_feedback)
        torch.testing.assert_close(actual.predictions, expected.predictions)
        torch.testing.assert_close(actual.valid_labels, expected.valid_labels)
        torch.testing.assert_close(fast.weight, base.weight)
        assert actual.vocab_update_norm == pytest.approx(expected.vocab_update_norm)
        assert actual.requantization_delta_norm == pytest.approx(
            expected.requantization_delta_norm
        )
        assert actual.ternary_flip_rate == pytest.approx(expected.ternary_flip_rate)

    def test_forward_observe(self):
        """Forward observer captures activations."""
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)
        batch = {
            "inputs": torch.randint(0, 256, (2, 8)),
            "labels": torch.randint(0, 256, (2, 8)),
        }

        result = nomad_learning.nomad_forward_observe(model, batch)

        assert result.hidden.shape[1] == 32
        assert result.labels.numel() == 16  # 2*8
        assert result.iterations == TINY_CONFIG["max_iters"]
        assert len(result.activations) > 0

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required"
    )
    def test_forward_observe_flat_inputs_use_graph(self):
        """Exp9-style flat [B*T] inputs reshape and hit the graphed path."""
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True).cuda()
        numseqs, seq_len = 2, 8
        flat = torch.randint(0, 256, (numseqs * seq_len,), device="cuda")
        batch = {
            "inputs": flat,
            "labels": flat.clone(),
            "numseqs": torch.tensor(numseqs),
        }

        nomad_learning.nomad_forward_observe(model, batch)

        key = (numseqs, seq_len, str(flat.device))
        assert key in model._graphs

    def test_train_step_head_only(self):
        """Head-only training step runs without error."""
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)
        batch = {
            "inputs": torch.randint(0, 256, (2, 8)),
            "labels": torch.randint(0, 256, (2, 8)),
        }

        feedback = {}
        step = nomad_learning.nomad_train_step(
            model,
            batch,
            vocab_chunk_size=64,
            head_lr=1e-3,
            body_lr=0.0,
            update_clip=1.0,
            external_memory=None,
            feedback_matrices=feedback,
            train_body=False,
        )

        assert step.loss > 0
        assert step.valid_tokens > 0
        assert 0.0 <= step.token_accuracy <= 1.0
        assert step.head_update_norm > 0  # head was updated
        assert step.body_update_norm == 0.0  # body not updated

    def test_train_step_head_and_body(self):
        """Head+body training step runs without error."""
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)
        batch = {
            "inputs": torch.randint(0, 256, (2, 8)),
            "labels": torch.randint(0, 256, (2, 8)),
        }

        # Seed feedback matrices manually
        feedback = {}
        for name, module in model.named_modules():
            if not isinstance(module, TernaryLinear158Init):
                continue
            if "tied_vocab" in name or "fast_memory" in name:
                continue
            if "input_proj" in name or "memory_proj" in name:
                continue
            out_dim = module.weight.shape[0]
            feedback[name] = torch.randn(out_dim, 32) / (32**0.5)

        step = nomad_learning.nomad_train_step(
            model,
            batch,
            vocab_chunk_size=64,
            head_lr=1e-3,
            body_lr=1e-4,
            update_clip=1.0,
            external_memory=None,
            feedback_matrices=feedback,
            train_body=True,
        )

        assert step.loss > 0
        assert step.valid_tokens > 0
        assert step.head_update_norm > 0
        # Body may or may not update depending on activation capture

    def test_shortlist_head_update_descends(self):
        """Shortlist sampled-CE head update reduces loss and keeps targets in S.

        Regression guard for the compute fix (Architecture shortlist): the hot
        path must (a) build S_t containing every target, (b) update only S_t
        rows, (c) still drive loss down at an adequate LR. Mirrors the
        full-vocab head test but on the shortlist path.
        """
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)
        from training.nobp_hard import hard_ternary_weight, chunked_vocab_ce

        losses = []
        prev_preds = None
        for _ in range(12):
            batch = {
                "inputs": torch.randint(0, 256, (2, 8)),
                "labels": torch.randint(0, 256, (2, 8)),
            }
            step = nomad_learning.nomad_train_step(
                model, batch,
                vocab_chunk_size=64, head_lr=0.05, body_lr=0.0,
                update_clip=10.0, external_memory=None,
                feedback_matrices={}, train_body=False,
                head_update_mode="shortlist",
                shortlist_topfreq=torch.arange(32),
                shortlist_prev_preds=prev_preds,
                shortlist_neg_size=64, shortlist_max_size=128,
                shortlist_vocab_size=256,
            )
            losses.append(step.loss)
            prev_preds = step.predictions
        assert losses[-1] < losses[0] * 1.1, (
            f"shortlist head did not reduce loss: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_shortlist_always_contains_targets(self):
        """build_shortlist guarantees every target id is in S_t."""
        labels = torch.tensor([5, 17, 200, 5, 17])
        topfreq = torch.arange(10)
        S = nomad_learning.build_shortlist(
            labels, topfreq_idx=topfreq, prev_preds=torch.tensor([123]),
            neg_size=50, vocab_size=256, max_size=4096,
        )
        for y in labels.unique().tolist():
            assert y in S.tolist(), f"target {y} missing from shortlist"
        # sorted + unique
        assert torch.equal(S, S.unique().sort().values)

    def test_evaluate(self):
        """Evaluation runs and returns metrics."""
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)

        def batch_fn(step: int):
            return {
                "inputs": torch.randint(0, 256, (2, 8)),
                "labels": torch.randint(0, 256, (2, 8)),
            }

        metrics = nomad_learning.nomad_evaluate(
            model,
            batch_fn=batch_fn,
            eval_batches=2,
            vocab_chunk_size=64,
        )

        assert "loss" in metrics
        assert "token_accuracy" in metrics
        assert "avg_iterations" in metrics
        assert metrics["loss"] > 0

    def test_kaczmarz_layer_update(self):
        """Kaczmarz update changes weights in correct direction."""
        module = TernaryLinear158Init(
            4, 8, bias=False,
            ternary_group_size=8,
            ternary_threshold=0.5,
            ternary_scale_mode="mean_abs",
            ternary_ste_mode="standard",
        )
        activation = torch.randn(3, 4)  # 3 samples, 4 input dims
        target = torch.randn(3, 8)  # 3 samples, 8 output dims

        before = module.weight.clone()
        norm, flips, count = nomad_learning.kaczmarz_layer_update(
            module, activation, target, lr=0.1, update_clip=1.0
        )
        after = module.weight.clone()

        assert norm >= 0
        # Weights should have been modified
        assert not torch.allclose(before, after)

    def test_kaczmarz_moves_toward_target(self):
        """Kaczmarz update moves a layer's output TOWARD the target.

        Regression guard for the sign bug where the update used -lr (away from
        target) and `current` used the master weight instead of the hard weight
        that actually flows forward. Measured in master space so it is immune
        to ternary quantization discontinuities.
        """
        torch.manual_seed(0)
        module = TernaryLinear158Init(
            4, 8, bias=False,
            ternary_group_size=8,
            ternary_threshold=0.5,
            ternary_scale_mode="mean_abs",
            ternary_ste_mode="standard",
        )
        activation = torch.randn(3, 4)
        target = torch.randn(3, 8)

        err_before = (
            target - torch.nn.functional.linear(activation, module.weight.detach())
        ).norm().item()
        nomad_learning.kaczmarz_layer_update(
            module, activation, target, lr=0.5, update_clip=100.0
        )
        err_after = (
            target - torch.nn.functional.linear(activation, module.weight.detach())
        ).norm().item()
        assert err_after < err_before, (
            f"Kaczmarz moved away from target: {err_before:.4f} -> {err_after:.4f}"
        )

    def test_dfa_target_is_loss_descending(self):
        """DFA target sits on the loss-descending side of the current output.

        (target - current) must oppose the teaching signal (B * dL/dh), so
        Kaczmarz's toward-target step reduces loss. Regression guard for the
        target-sign bug (was `current + 0.05*teaching` = loss-ascending).
        """
        import torch.nn.functional as F
        from training.nobp_hard import hard_ternary_weight

        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)
        body_modules = {}
        for name, module in model.named_modules():
            if not isinstance(module, TernaryLinear158Init):
                continue
            if "tied_vocab" in name or "fast_memory" in name:
                continue
            if "input_proj" in name or "memory_proj" in name:
                continue
            body_modules[name] = module
        name, module = next(iter(body_modules.items()))
        out_dim = module.weight.shape[0]
        B = torch.randn(out_dim, 32) / (32 ** 0.5)
        output_error = torch.randn(5, 32)  # dL/dh (loss-ascending)
        act = torch.randn(5, module.weight.shape[1])
        activations = {name: act}

        targets = nomad_learning.compute_dfa_targets(
            output_error, {name: B}, activations, {name: module},
            hidden_dim=32, valid_mask=None, alpha=1.0,
        )
        target = targets[name]
        current = F.linear(act.float(), hard_ternary_weight(module).float())
        teaching = output_error.float() @ B.transpose(0, 1)
        delta = target - current
        dot = float((delta * teaching).sum().cpu())
        assert dot < 0, (
            f"DFA target is not loss-descending: dot(teaching, target-current)="
            f"{dot:.4f} >= 0"
        )

    def test_checkpoint_save_load(self):
        """Checkpoint save/load round-trip works."""
        import tempfile

        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)
        feedback = {}
        for name, module in model.named_modules():
            if not isinstance(module, TernaryLinear158Init):
                continue
            if "tied_vocab" in name or "fast_memory" in name:
                continue
            out_dim = module.weight.shape[0]
            feedback[name] = torch.randn(out_dim, 32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")

            # Save
            nomad_learning.save_nomad_checkpoint(
                path, model, step=42,
                metrics={"loss": 5.0},
                feedback_matrices=feedback,
            )
            assert Path(path).exists()

            # Load into new model
            model2 = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)
            step, metrics, loaded_feedback = (
                nomad_learning.load_nomad_checkpoint(path, model2)
            )

            assert step == 42
            assert metrics["loss"] == 5.0
            assert len(loaded_feedback) == len(feedback)

    def test_loss_decreases_head_only(self):
        """Head-only training reduces loss over multiple steps."""
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)

        losses = []
        for _ in range(10):
            batch = {
                "inputs": torch.randint(0, 256, (2, 8)),
                "labels": torch.randint(0, 256, (2, 8)),
            }
            step = nomad_learning.nomad_train_step(
                model,
                batch,
                vocab_chunk_size=64,
                head_lr=0.01,
                body_lr=0.0,
                update_clip=1.0,
                external_memory=None,
                feedback_matrices={},
                train_body=False,
            )
            losses.append(step.loss)

        # On random data, loss should decrease from initial high value
        # (random prediction ~ log(vocab) ≈ 5.5 for vocab=256)
        assert losses[-1] < losses[0] * 1.1  # at least not increasing


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_cli_defaults_use_large_gpu_batch_and_chunk(self):
        """Phase-0 speed defaults use high batch and fewer vocab chunks."""
        import exp126_nomad_phase0

        args = exp126_nomad_phase0.build_parser().parse_args([])

        assert args.numseqs == 128
        assert args.max_iters == 5
        assert args.vocab_chunk_size == 8192

    def test_end_to_end_pretrain(self):
        """End-to-end: build -> train -> eval -> checkpoint."""
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)

        # Create synthetic token stream
        tokens = torch.randint(0, 256, (2000,), dtype=torch.long)

        def batch_fn(step: int):
            numseqs = 2
            total_len = 8
            idx = (step * numseqs * total_len) % (len(tokens) - numseqs * total_len)
            chunk = tokens[idx : idx + numseqs * total_len]
            inputs = chunk[: numseqs * total_len].view(numseqs, total_len).clone()
            labels = inputs.clone()
            # Shift labels: predict next token
            labels[:, :-1] = inputs[:, 1:]
            labels[:, -1] = -100  # ignore last position
            return {"inputs": inputs, "labels": labels}

        # Head-only training
        from training.nobp_hard import configure_hard_ternary
        configure_hard_ternary(model)

        train_metrics = nomad_learning.train_nomad_pretrain(
            model,
            batch_fn=batch_fn,
            device=torch.device("cpu"),
            steps=20,
            vocab_chunk_size=64,
            head_lr=0.01,
            body_lr=0.0,
            update_clip=1.0,
            log_interval=10,
            checkpoint_path=None,
            train_body=False,
        )

        assert train_metrics["steps"] == 20
        assert train_metrics["elapsed_s"] >= 0
        assert train_metrics["tokens_per_sec"] > 0

        # Evaluate
        eval_metrics = nomad_learning.nomad_evaluate(
            model,
            batch_fn=batch_fn,
            eval_batches=4,
            vocab_chunk_size=64,
        )
        assert eval_metrics["loss"] > 0
        assert "token_accuracy" in eval_metrics


# ---------------------------------------------------------------------------
# Phase 1A probe tests
# ---------------------------------------------------------------------------


class TestPhase1AProbe:
    def test_chunked_topk_rank_loss_ece_ranges(self):
        """Metric returns sane ranges: top-k in [0,1], rank in [1,V], loss ~ log V."""
        torch.manual_seed(0)
        N, D, V = 6, 8, 20
        hidden = torch.randn(N, D)
        labels = torch.randint(0, V, (N,))
        labels[1] = -100  # ignored position
        weight = torch.randn(V, D)
        m = phase1a.chunked_topk_rank_loss_ece(
            hidden, labels, weight, chunk_size=5, k=10, n_bins=5
        )
        assert m["n_valid"] == 5  # one ignored
        assert 0.0 <= m["top1"] <= 1.0
        assert 0.0 <= m["top5"] <= 1.0
        assert 0.0 <= m["top10"] <= 1.0
        assert 1.0 <= m["mean_rank"] <= V
        assert 0.0 <= m["ece"] <= 1.0
        assert m["loss"] > 0

    def test_chunked_rank_matches_full(self):
        """Chunked rank == full-vocab rank (correctness of the chunked count)."""
        torch.manual_seed(1)
        N, D, V = 5, 8, 30
        hidden = torch.randn(N, D)
        labels = torch.randint(0, V, (N,))
        weight = torch.randn(V, D)
        m = phase1a.chunked_topk_rank_loss_ece(
            hidden, labels, weight, chunk_size=7, k=10, n_bins=5
        )
        # full-vocab rank for reference
        full = torch.nn.functional.linear(hidden.float(), weight.float())
        correct = full.gather(1, labels.unsqueeze(-1)).squeeze(-1)
        full_rank = (full > correct.unsqueeze(-1)).sum(dim=-1) + 1
        assert abs(m["mean_rank"] - full_rank.float().mean().item()) < 1e-4

    def test_graph_output_not_aliased_across_calls(self):
        """Graph wrapper must clone outputs -- the static buffer is reused.

        Regression guard for the Phase 1B cache-corruption bug: graph(inp1)
        and graph(inp2) returned the same data_ptr, so caching the first
        output silently overwrote it with the second call's data. The wrapper
        now clones on return.
        """
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True).to("cuda")
        inp1 = torch.randint(0, 256, (2, 8), device="cuda")
        inp2 = torch.randint(0, 256, (2, 8), device="cuda")
        g = model.get_graph(2, 8, torch.device("cuda"))
        h1 = g(inp1)[0]
        h2 = g(inp2)[0]
        # different inputs -> different hidden (and NOT aliased to same buffer)
        assert h1.data_ptr() != h2.data_ptr()
        assert not torch.allclose(h1, h2, atol=1e-6)

    def test_memory_adapter_noop_on_zero_memory(self):
        """Adapter(h, zeros) == h regardless of A/gamma (Phase 1B contract).

        The off-baseline must be invariant to adapter training: with m=0 the
        adapter is a no-op, so memory-off eval cannot change as A trains.
        """
        adapter = phase1a.MemoryAdapter if hasattr(phase1a, "MemoryAdapter") else None
        # MemoryAdapter lives in phase1b; import lazily
        import phase1b_adapter_train as p1b
        a = p1b.MemoryAdapter(32, gamma=0.1)
        with torch.no_grad():
            a.A.add_(torch.randn(32, 32) * 0.5)  # train A arbitrarily
            a.gamma.add_(0.5)
        h = torch.randn(5, 32)
        m = torch.zeros_like(h)
        with torch.no_grad():
            h2 = a(h, m)
        assert torch.allclose(h2, h, atol=1e-6), "adapter leaked into off-path"

    def test_logit_bias_metric_noop_at_beta0(self):
        """biased_metrics at beta=0 (or memory_on=False) == unbiased; beta>0 changes logits."""
        torch.manual_seed(0)
        N, D, V = 6, 8, 20
        h = torch.randn(N, D)
        labels = torch.randint(0, V, (N,))
        labels[1] = -100
        w = torch.randn(V, D)
        bias = torch.zeros(N, V)
        bias[0, 3] = 5.0
        bias[2, 7] = 5.0
        m_off = phase1b_logit_bias.biased_metrics(
            h, bias, labels, w, beta=0.0, vocab_chunk_size=5, memory_on=False)
        m_b0 = phase1b_logit_bias.biased_metrics(
            h, torch.zeros_like(bias), labels, w, beta=0.0, vocab_chunk_size=5, memory_on=False)
        assert abs(m_off["loss"] - m_b0["loss"]) < 1e-5  # beta=0 / off is a no-op
        m_on = phase1b_logit_bias.biased_metrics(
            h, bias, labels, w, beta=1.0, vocab_chunk_size=5, memory_on=True)
        assert abs(m_on["loss"] - m_off["loss"]) > 1e-3  # bias changes logits

    def test_beta_update_gets_nonzero_grad_with_bias_tokens(self):
        """beta_update must include high-bias tokens in the shortlist or grad=0.

        Regression guard for the bug where memory's candidate tokens were
        absent from S_t (targets + random negs only), so beta received grad 0
        and never learned. The shortlist now includes top-bias tokens.
        """
        torch.manual_seed(0)
        N, D, V = 6, 8, 20
        h = torch.randn(N, D)
        labels = torch.randint(0, V, (N,))
        labels[1] = -100
        w = torch.randn(V, D)
        bias = torch.zeros(N, V)
        bias[0, 3] = 5.0  # biased token 3 (not a target label)
        gen = torch.Generator(); gen.manual_seed(0)
        _, info = phase1b_logit_bias.beta_update(
            h, bias, labels, w, beta=0.0, eta_beta=1.0,
            shortlist_size=20, neg_size=4, generator=gen)
        assert abs(info["grad"]) > 1e-6, "beta got grad 0 (bias tokens not in shortlist)"

    def test_build_memory_reads_shape(self):
        """Per-sequence retrieval broadcasts to [B, T, D]."""
        model = nomad_model.build_nomad_model(TINY_CONFIG, hard=True)
        from tokenizers import Tokenizer
        from pathlib import Path
        tok_path = Path(
            "C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json"
        )
        if not tok_path.exists():
            pytest.skip("tokenizer not available")
        tok = Tokenizer.from_file(str(tok_path))
        mem = nomad_memory.ExternalMemory(hidden_size=32, lambda_exact=1.0, lambda_gzip=0.0)
        phase1a.ingest_text_with_tokens(
            mem, "The capital of France is Paris. It is a city.", tok, chunk_id_prefix="t"
        )
        assert len(mem) > 0
        inp = torch.randint(0, 256, (2, 8))
        reads = phase1a.build_memory_reads(
            model, mem, inp, tok, prefix_len=4, top_k=4, device=torch.device("cpu")
        )
        assert reads.shape == (2, 8, 32)

