import torch
import torch.nn as nn
import numpy as np
from encoder import VQVAEEncoder
from decoder import VQVAEDecoder
from quantizer import VectorQuantizer

class VQVAE(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        self.encoder = VQVAEEncoder(embedding_dim=embedding_dim)
        self.vector_quantization = VectorQuantizer(
            num_embeddings=n_embeddings, embedding_dim=embedding_dim, beta=beta
        )
        self.decoder = VQVAEDecoder(embedding_dim=embedding_dim)
        
        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None
    
    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)
        
        return embedding_loss, x_hat, perplexity