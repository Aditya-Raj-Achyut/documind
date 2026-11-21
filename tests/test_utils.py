from app.utils import clean_text, extract_json, rrf, split_text, tokenize


def _sample(n: int = 200) -> str:
    return " ".join(f"Sentence number {i} talks about something important." for i in range(n))


def test_chunks_never_exceed_size():
    chunks = split_text(_sample(), size=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_chunks_overlap():
    chunks = split_text(_sample(), size=300, overlap=60)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert " ".join(nxt.split()[:3]) in prev


def test_long_sentence_is_hard_split():
    chunks = split_text("word " * 1000, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_no_text_lost():
    text = _sample(50)
    chunks = split_text(text, size=300, overlap=60)
    for i in range(50):
        assert any(f"Sentence number {i} " in c for c in chunks)


def test_rrf_prefers_items_found_by_both_retrievers():
    assert rrf([[1, 2], [3, 2]])[0] == 2
    assert rrf([[1, 2], [1, 3]])[0] == 1


def test_clean_text():
    assert clean_text("inter-\nnational  law\nis   fun") == "international law is fun"


def test_tokenize():
    assert tokenize("Hello, World 2024!") == ["hello", "world", "2024"]


def test_extract_json():
    assert extract_json('Sure! {"a": 1} hope it helps') == {"a": 1}
    assert extract_json("no json here") is None
    assert extract_json("{broken") is None
