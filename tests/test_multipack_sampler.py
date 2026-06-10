import numpy as np
import pytest

pytest.importorskip("numba")

from multipack_sampler import lpt_check, lpt_with_result


def test_lpt_check_accepts_balanced_lengths():
    heap = np.zeros(8, dtype=np.int64)
    lengths = np.array([10, 10, 10, 10], dtype=np.int64)
    assert lpt_check(heap, lengths, c=15, n=4) is True


def test_lpt_check_rejects_oversized_batch():
    heap = np.zeros(8, dtype=np.int64)
    lengths = np.array([20, 20, 20], dtype=np.int64)
    assert lpt_check(heap, lengths, c=15, n=3) is False


def test_lpt_with_result_is_deterministic_for_rank():
    heap = np.zeros(8, dtype=np.int64)
    lengths = np.array([12, 8, 6, 4], dtype=np.int64)
    first = lpt_with_result(heap.copy(), lengths, n=2, rank=0)
    second = lpt_with_result(heap.copy(), lengths, n=2, rank=0)
    assert np.array_equal(first, second)
