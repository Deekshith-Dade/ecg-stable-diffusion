from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class QuantizeOutput:
    z_q: torch.Tensor
    perplexity: torch.Tensor
    # only contains 'commitment_loss' and 'codebook_loss'
    quantize_losses: dict[str, torch.Tensor]
    encodings: torch.Tensor
    encoding_indices: torch.Tensor


@dataclass
class EncodeOutput:
    quantize_output: QuantizeOutput


@dataclass
class DecodeOutput:
    x_hat: torch.Tensor


@dataclass
class ForwardOutput:
    x_hat: torch.Tensor
    z_q: torch.Tensor
    # only contains 'commitment_loss' and 'codebook_loss'
    perplexity: torch.Tensor
    quantize_losses: dict[str, torch.Tensor]
    encoding_indices: torch.Tensor


class VQVAEBase(nn.Module):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> EncodeOutput:
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> DecodeOutput:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, verbose: bool = False) -> ForwardOutput:
        pass
