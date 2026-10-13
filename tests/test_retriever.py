from src.state.models import Chunk
from src.retrieval.embedder import embed_texts
from src.retrieval.index import build_index
from src.retrieval.retriever import retrieve_chunks


def test_retrieve_chunks():
    chunks = [
        Chunk(
            chunk_id="chunk_1",
            chunk_text="Proof of Work requires computational work.",
            document_id="doc1"
        ),
        Chunk(
            chunk_id="chunk_2",
            chunk_text="Proof of Stake selects validators based on stake.",
            document_id="doc2"
        ),
        Chunk(
            chunk_id="chunk_3",
            chunk_text="Blockchain stores transactions in linked blocks.",
            document_id="doc3"
        )
    ]

    texts = [chunk.chunk_text for chunk in chunks]

    embeddings = embed_texts(texts)
    index = build_index(embeddings)

    query = "Why does Proof of Work require computation?"

    retrieved_chunks, scores = retrieve_chunks(
        query=query,
        index=index,
        chunks=chunks,
        k=2
    )

    assert len(retrieved_chunks) == 2
    assert all(isinstance(chunk, Chunk) for chunk in retrieved_chunks)

    retrieved_ids = [chunk.chunk_id for chunk in retrieved_chunks]

    assert "chunk_1" in retrieved_ids

def test_retrieve_chunks_when_k_exceeds_chunk_count():
    chunks = [
        Chunk(
            chunk_id="chunk_1",
            chunk_text="Proof of Work requires computational work.",
            document_id="doc1"
        ),
        Chunk(
            chunk_id="chunk_2",
            chunk_text="Proof of Stake selects validators based on stake.",
            document_id="doc2"
        ),
        Chunk(
            chunk_id="chunk_3",
            chunk_text="Blockchain stores transactions in linked blocks.",
            document_id="doc3"
        )
    ]

    texts = [chunk.chunk_text for chunk in chunks]

    embeddings = embed_texts(texts)
    index = build_index(embeddings)

    retrieved_chunks, scores = retrieve_chunks(
        query="Explain blockchain consensus mechanisms.",
        index=index,
        chunks=chunks,
        k=5
    )

    assert len(retrieved_chunks) == 3
    assert all(isinstance(chunk, Chunk) for chunk in retrieved_chunks)
    assert all(chunk.chunk_id in {
        "chunk_1",
        "chunk_2",
        "chunk_3"
    } for chunk in retrieved_chunks)