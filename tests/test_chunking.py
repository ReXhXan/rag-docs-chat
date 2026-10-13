import pytest

from app.core.chunking import chunk_pages, split_text


def test_short_text_single_chunk():
    assert split_text("hello world", 100, 10) == ["hello world"]


def test_empty_text():
    assert split_text("   ", 100, 10) == []


def test_chunks_respect_size_and_cover_text():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert "word0" in chunks[0] and "word499" in chunks[-1]


def test_overlap_carries_context():
    text = "\n\n".join(f"Paragraph {i} " + "x" * 80 for i in range(6))
    chunks = split_text(text, chunk_size=200, overlap=50)
    for a, b in zip(chunks, chunks[1:]):
        assert a[-20:] in b  # tail of previous chunk appears in next


def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("abc", 10, 10)


def test_chunk_metadata_and_stable_ids():
    pages = [("a " * 300, {"source": "f.pdf", "page": 3})]
    c1, c2 = chunk_pages(pages, 100, 10), chunk_pages(pages, 100, 10)
    assert [c.id for c in c1] == [c.id for c in c2]
    assert c1[0].metadata == {"source": "f.pdf", "page": 3, "chunk_index": 0}
