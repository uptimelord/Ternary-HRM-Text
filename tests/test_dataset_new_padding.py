def _pad_token_id(tokenizer_info: dict) -> int:
    return int(tokenizer_info.get("pad_token_id", 0))


def test_pad_token_defaults_to_zero():
    assert _pad_token_id({}) == 0


def test_pad_token_reads_metadata_override():
    assert _pad_token_id({"pad_token_id": 3}) == 3
