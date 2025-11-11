from torch import nn
import torch
import numpy as np
from typing import Tuple


class SparseAutoencoder(nn.Module):
    """Two‑layer sparse auto‑encoder for 512‑D latent vectors with an over‑complete
    hidden "concept" layer. The decoder weights constitute the learned
    dictionary atoms.
    """

    def __init__(
        self,
        input_dim: int = 512,
        code_dim: int = 2048,
        activation: nn.Module = nn.ReLU(),
        tied_weights: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, code_dim, bias=True)
        self.decoder = nn.Linear(code_dim, input_dim, bias=True)
        self.activation = activation
        if tied_weights:
            # tie decoder weights to encoder weights (transpose)
            self.decoder.weight = nn.Parameter(self.encoder.weight.t())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.activation(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.activation(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.decoder(z)

    def concept_vector(self, concept_idx: int) -> torch.Tensor:
        return self.decoder.weight[:, concept_idx].detach().clone()
