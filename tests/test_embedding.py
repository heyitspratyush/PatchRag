
import torch

from src.retrieval.embedder import mean_pooling
from src.retrieval.embedder import embed_text
from src.retrieval.embedder import embed_texts

def test_mean_pooling_ignores_padding():
    last_hidden_state = torch.tensor([
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [100.0, 100.0]
        ]
    ])

    attention_mask = torch.tensor([
        [1, 1, 1, 0]
    ])

    result = mean_pooling(
        last_hidden_state,
        attention_mask
    )

    expected = torch.tensor([
        [3.0, 4.0]
    ])

    assert torch.allclose(result, expected)

def test_mean_pooling_output_shape():
    last_hidden_state = torch.tensor([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ],
        [
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0]
        ]
    ])

    attention_mask = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 0, 0]
    ])

    result = mean_pooling(
        last_hidden_state,
        attention_mask
    )

    assert result.shape == (2, 3)




def test_embed_text_output_shape():
    text = "Proof of Work uses computational work."

    embedding = embed_text(text)

    assert embedding.shape == (1, 384)

from src.retrieval.embedder import embed_texts


def test_embed_texts_output_shape():
    texts = [
        "Proof of Work uses computational work.",
        "Proof of Stake uses stake.",
        "Blockchain stores linked blocks."
    ]

    embeddings = embed_texts(texts)

    assert embeddings.shape == (3, 384)

from src.retrieval.embedder import normalize_embeddings


def test_normalize_embeddings():
    embeddings = torch.tensor([
        [3.0, 4.0],
        [5.0, 12.0]
    ])

    normalized = normalize_embeddings(embeddings)

    expected = torch.tensor([
        [0.6, 0.8],
        [0.38461538, 0.92307692]
    ])

    assert torch.allclose(normalized, expected)

def test_embed_texts_are_normalized():
    texts = [
        "Proof of Work uses computational work.",
        "Proof of Stake uses stake."
    ]

    embeddings = embed_texts(texts)

    norms = torch.linalg.vector_norm(embeddings, dim=1)

    assert torch.allclose(
        norms,
        torch.ones(2),
        atol=1e-6
    )