import sys
sys.path.append("..")  

import torch.nn as nn
from vq_vae.encoder import VQVAEEncoder
from vq_vae.decoder import VQVAEDecoder
from vq_vae.quantizer import VectorQuantizer

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

class VQVAE(nn.Module):
    def __init__(self, model_config):
        super(VQVAE, self).__init__()

        self.in_channels = model_config["in_channels"]
        self.dim = model_config["init_dim"]
        self.attn = model_config["attention"]
        
        self.n_embeddings = model_config["codebook_size"]
        self.embedding_dim = model_config["embedding_dim"]
        self.beta = model_config["beta"]
        
        
        dim_mults = model_config["dim_mults"]
        dims = [self.dim, *map(lambda m: self.dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.norm_channels = model_config["norm_channels"]


        self.encoder_conv_in = nn.Conv2d(self.in_channels, dims[0], kernel_size=1, padding=0)
        self.encoder = VQVAEEncoder(in_out, attn=self.attn)
        
        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, dims[-1])
        self.encoder_conv_out = nn.Conv2d(dims[-1], self.embedding_dim, kernel_size=(1, 3), padding=(0, 1))
        
        self.pre_quant_conv = nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=1, padding=0)
        
        
        self.vector_quantization = VectorQuantizer(
            num_embeddings=self.n_embeddings, embedding_dim=self.embedding_dim, beta=self.beta
        )
        
        
        self.post_quant_conv = nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=1, padding=0)
        self.decoder_conv_in = nn.Conv2d(self.embedding_dim, dims[-1], kernel_size=(1, 3), padding=(0, 1))
        
        self.decoder = VQVAEDecoder(in_out, attn=self.attn)
        
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, dims[0])
        self.decoder_conv_out = nn.Conv2d(dims[0], self.in_channels, kernel_size=1, padding=0)

    def encode(self, x):
        x = self.encoder_conv_in(x)
        z_e = self.encoder(x)
        
        z_e = self.encoder_norm_out(z_e)
        z_e = nn.SiLU()(z_e)
        z_e = self.encoder_conv_out(z_e)
        
        z_e = self.pre_quant_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        
        return embedding_loss, z_q, perplexity

    def decode(self, z_q):
        out = z_q
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        
        out = self.decoder(out)
        
        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out

    def forward(self, x, verbose=False):
        embedding_loss, z_q, perplexity = self.encode(x)
        x_hat = self.decode(z_q)
        if verbose:
            print(f"Embedding Loss: {embedding_loss.item()}, Perplexity: {perplexity.item()}")
        return embedding_loss, x_hat, perplexity