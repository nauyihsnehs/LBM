from dataclasses import field
from typing import List

from pydantic.dataclasses import dataclass

from ..base import BaseConditionerConfig


@dataclass
class LightParamsEmbedderConfig(BaseConditionerConfig):
    """
    Configuration for the light params embedder.

    Args:
        input_key (str): Key for the light params input.
        num_frequencies (int): Number of Fourier frequencies per parameter.
        include_input (bool): Whether to include raw inputs in the embedding.
        hidden_dims (List[int]): Hidden layer sizes for the MLP.
        num_tokens (int): Number of tokens to output (SD1.5 uses 77).
        embedding_dim (int): Embedding dimension per token (SD1.5 uses 768).
    """

    input_key: str = "light_params"
    num_frequencies: int = 6
    include_input: bool = True
    hidden_dims: List[int] = field(default_factory=lambda: [256, 512])
    num_tokens: int = 77
    embedding_dim: int = 768

    def __post_init__(self):
        super().__post_init__()
        assert self.num_frequencies > 0, "num_frequencies must be positive"
        assert self.num_tokens > 0, "num_tokens must be positive"
        assert self.embedding_dim > 0, "embedding_dim must be positive"
