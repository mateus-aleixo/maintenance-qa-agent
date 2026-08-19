from conformal_rag.ingest import _find_cut, chunk_pages, clean_page


def test_clean_page_drops_furniture():
    raw = "TM 9-8000\nThe engine block houses the cylinders.\n42\n  spaced   text  "
    cleaned = clean_page(raw)
    assert "TM 9-8000" not in cleaned
    assert "42" not in cleaned.splitlines()
    assert "The engine block houses the cylinders." in cleaned
    assert "spaced text" in cleaned


def test_chunks_cover_text_and_carry_pages():
    sentence = "The quick brown fox inspects the manifold. "
    pages = [(1, sentence * 30), (2, sentence * 30)]
    chunks = list(chunk_pages("doc.pdf", pages, chunk_chars=400, overlap=80))
    assert len(chunks) > 2
    assert chunks[0].page == 1
    assert chunks[-1].page == 2
    assert all(c.text for c in chunks)
    # overlap: consecutive chunks share content
    assert chunks[0].text[-40:].split()[0] in chunks[1].text


def test_cut_prefers_sentence_boundary():
    text = "First sentence here. Second sentence continues onwards without end"
    cut = _find_cut(text, 40)
    assert text[:cut].rstrip().endswith(".")


def test_overlap_must_be_smaller():
    import pytest

    with pytest.raises(ValueError):
        list(chunk_pages("d", [(1, "x" * 100)], chunk_chars=100, overlap=100))
