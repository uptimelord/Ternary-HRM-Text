import pytest


def test_fetch_next_yields_overflow_and_returns_valid_prompt():
    """Contract for inference_generate's nested fetch_next helper."""

    class _Ckpt:
        def __init__(self, sizes: list[int]):
            self._sizes = sizes

        def tokenize_prompt(self, _cond, _prompt):
            return type("Tok", (), {"size": self._sizes.pop(0)})()

    def fetch_next(iterator, ckpt, max_tokens):
        for pid, p_tuple in iterator:
            tok = ckpt.tokenize_prompt(*p_tuple)
            if tok.size >= max_tokens:
                yield pid, ""
            else:
                return pid, tok
        return -1, None

    overflow = iter([(1, ("c", "long"))])
    gen = fetch_next(overflow, _Ckpt([8]), max_tokens=4)
    pid, text = next(gen)
    assert pid == 1 and text == ""

    ok = iter([(2, ("c", "ok"))])
    gen_ok = fetch_next(ok, _Ckpt([2]), max_tokens=4)
    with pytest.raises(StopIteration) as excinfo:
        next(gen_ok)
    pid, tok = excinfo.value.value
    assert pid == 2 and tok is not None
