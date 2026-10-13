from src.retrieval.embedder import embed_texts
from src.retrieval.index import build_index, search_index
import torch
import numpy as np
import pytest

from src.state.models import Chunk
from src.retrieval.index import map_indices_to_chunks


def test_build_index():
    texts = [
        "Proof of Work uses computational work.",
        "Proof of Stake uses stake.",
        "Blockchain stores linked blocks."
    ]

    embeddings = embed_texts(texts)

    index = build_index(embeddings)

    assert index.ntotal == 3

def test_search_index():
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32
    )

    index = build_index(embeddings)

    query_vector = torch.tensor(
        [[0.0, 1.0, 0.0]],
        dtype=torch.float32
    )

    scores, indices = search_index(
        index,
        query_vector,
        k=2
    )

    assert indices[0][0] == 1
    assert abs(scores[0][0] - 1.0) < 1e-6
    assert indices.shape == (1, 2)

def test_map_indices_to_chunks():
    chunks = [
        Chunk(
            chunk_id="chunk_A",
            chunk_text="Blockchain stores records.",
            document_id="doc1"
        ),
        Chunk(
            chunk_id="chunk_B",
            chunk_text="Proof of Work uses computation.",
            document_id="doc2"
        ),
        Chunk(
            chunk_id="chunk_C",
            chunk_text="Proof of Stake uses stake.",
            document_id="doc3"
        )
    ]

    indices = np.array([[1, 2]])

    retrieved_chunks = map_indices_to_chunks(
        indices,
        chunks
    )

    assert len(retrieved_chunks) == 2
    assert retrieved_chunks[0].chunk_id == "chunk_B"
    assert retrieved_chunks[1].chunk_id == "chunk_C"


def test_map_indices_to_chunks_skips_missing_positions():
    chunks = [
        Chunk(
            chunk_id="chunk_A",
            chunk_text="Blockchain stores records.",
            document_id="doc1"
        ),
        Chunk(
            chunk_id="chunk_B",
            chunk_text="Proof of Work uses computation.",
            document_id="doc2"
        )
    ]

    indices = np.array([[1, -1]])

    retrieved_chunks = map_indices_to_chunks(
        indices,
        chunks
    )

    assert len(retrieved_chunks) == 1
    assert retrieved_chunks[0].chunk_id == "chunk_B"


def test_map_indices_to_chunks_rejects_invalid_position():
    chunks = [
        Chunk(
            chunk_id="chunk_A",
            chunk_text="Blockchain stores records.",
            document_id="doc1"
        )
    ]

    indices = np.array([[2]])

    with pytest.raises(ValueError):
        map_indices_to_chunks(indices, chunks)