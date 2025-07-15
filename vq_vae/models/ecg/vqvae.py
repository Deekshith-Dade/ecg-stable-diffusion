from .quantizer import VectorQuantizer
from .decoder import VQVAEDecoder
from .encoder import VQVAEEncoder
from ..vqvae_base import VQVAEBase, EncodeOutput, DecodeOutput, ForwardOutput
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# model_config = {
#     "dim_mults": (1, 2, 4),
#     "in_channels": 1,
#     "init_dim": 128,
#     "embedding_dim": 8,
#     "codebook_size": 1024,
#     "beta": 0.25,
#     "attention": False,
#     "norm_channels": 32,
# }


class VQVAEECG(VQVAEBase):
    def __init__(self, model_config):
        super().__init__()

        self.in_channels = model_config["in_channels"]
        self.dim = model_config["init_dim"]
        self.attn = model_config["attention"]

        self.codebook_size = model_config["codebook_size"]
        self.embedding_dim = model_config["embedding_dim"]
        self.beta = model_config["beta"]

        dim_mults = model_config["dim_mults"]
        dims = [self.dim, *map(lambda m: self.dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.norm_channels = model_config["norm_channels"]

        self.encoder_conv_in = nn.Conv2d(
            self.in_channels, dims[0], kernel_size=1, padding=0)
        self.encoder = VQVAEEncoder(in_out, attn=self.attn)

        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, dims[-1])
        self.encoder_conv_out = nn.Conv2d(
            dims[-1], self.embedding_dim, kernel_size=(1, 3), padding=(0, 1))

        self.pre_quant_conv = nn.Conv2d(
            self.embedding_dim, self.embedding_dim, kernel_size=1, padding=0)

        self.vector_quantization = VectorQuantizer(
            num_embeddings=self.codebook_size, embedding_dim=self.embedding_dim
        )

        self.post_quant_conv = nn.Conv2d(
            self.embedding_dim, self.embedding_dim, kernel_size=1, padding=0)
        self.decoder_conv_in = nn.Conv2d(
            self.embedding_dim, dims[-1], kernel_size=(1, 3), padding=(0, 1))

        self.decoder = VQVAEDecoder(in_out, attn=self.attn)

        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, dims[0])
        self.decoder_conv_out = nn.Conv2d(
            dims[0], self.in_channels, kernel_size=1, padding=0)

    def encode(self, x) -> EncodeOutput:
        x = self.encoder_conv_in(x)
        z_e = self.encoder(x)

        z_e = self.encoder_norm_out(z_e)
        z_e = nn.SiLU()(z_e)
        z_e = self.encoder_conv_out(z_e)

        z_e = self.pre_quant_conv(z_e)
        quant_output = self.vector_quantization(z_e)

        return EncodeOutput(quant_output)

    def decode(self, z_q) -> DecodeOutput:
        out = z_q
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)

        out = self.decoder(out)

        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return DecodeOutput(out)

    def forward(self, x, verbose=False) -> ForwardOutput:
        encode_output = self.encode(x)
        x_hat = self.decode(encode_output.quantize_output.z_q).x_hat
        if verbose:
            print(
                f"Perplexity: {encode_output.quantize_output.perplexity.item()}")
        return ForwardOutput(x_hat, encode_output.quantize_output.z_q, encode_output.quantize_output.perplexity, encode_output.quantize_output.quantize_losses, encode_output.quantize_output.encoding_indices)
