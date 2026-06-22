import pytest
import torch

from training.sdm_buffer import HeldOutTraceError, SparseDistributedMemory, UnverifiedTraceError


def _memory():
    return SparseDistributedMemory(
        address_bits=64,
        data_size=16,
        num_hard_locations=512,
        k_active=64,
        seed=4,
        held_out_ids={"held"},
    )


def test_exact_address_recovers_written_vector():
    memory = _memory()
    address = torch.randint(0, 2, (1, 64), dtype=torch.bool)
    data = torch.randn(1, 16)
    memory.write(address, data, verified=True, task_ids=["train-1"])
    recovered = memory.read(address)
    assert torch.allclose(recovered, data, atol=1e-6)


def test_noisy_address_retrieves_same_trace():
    memory = _memory()
    address = torch.randint(0, 2, (1, 64), dtype=torch.bool)
    data = torch.randn(1, 16)
    memory.write(address, data, verified=True, task_ids=["train-2"])
    noisy = address.clone()
    noisy[:, :2] = ~noisy[:, :2]
    recovered = memory.read(noisy)
    cosine = torch.nn.functional.cosine_similarity(recovered, data).item()
    assert cosine > 0.8


def test_unverified_and_held_out_writes_are_refused():
    memory = _memory()
    address = torch.randint(0, 2, (1, 64), dtype=torch.bool)
    data = torch.randn(1, 16)
    with pytest.raises(UnverifiedTraceError):
        memory.write(address, data, verified=False, task_ids=["train-3"])
    with pytest.raises(HeldOutTraceError):
        memory.write(address, data, verified=True, task_ids=["held"])
    assert memory.write_count == 0


def test_memory_is_cpu_resident_and_requires_cpu_inputs():
    memory = _memory()
    assert memory.hard_locations.device.type == "cpu"
    assert memory.counters.device.type == "cpu"


def test_save_load_preserves_memory(tmp_path):
    memory = _memory()
    address = torch.randint(0, 2, (1, 64), dtype=torch.bool)
    data = torch.randn(1, 16)
    memory.write(address, data, verified=True, task_ids=["train-4"])
    path = memory.save(tmp_path / "sdm.pt")
    restored = SparseDistributedMemory.load(path, held_out_ids={"held"})
    assert restored.task_ids == ["train-4"]
    assert restored.write_count == 1
    assert torch.allclose(restored.read(address), data, atol=1e-6)
