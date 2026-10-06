
import torch

from src.retrieval.embedder import mean_pooling


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