from src.retrieval.chunker import chunk_text


def test_chunk_text():
    document_text = "one two three four five six seven eight nine ten"

    chunks = chunk_text(document_text, "doc1")

    assert len(chunks) == 3

    assert chunks[0].chunk_text == "one two three four five"
    assert chunks[1].chunk_text == "four five six seven eight"
    assert chunks[2].chunk_text == "seven eight nine ten"

def test_chunk_ids_are_deterministic():
    document_text = "one two three four five six seven eight nine ten"

    chunks_first = chunk_text(document_text, "doc1")
    chunks_second = chunk_text(document_text, "doc1")

    assert chunks_first[0].chunk_id == chunks_second[0].chunk_id
    assert chunks_first[1].chunk_id == chunks_second[1].chunk_id
    assert chunks_first[2].chunk_id == chunks_second[2].chunk_id

def test_chunk_ids_are_unique():
    document_text = "one two three four five six seven eight nine ten"

    chunks = chunk_text(document_text, "doc1")

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))