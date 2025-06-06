import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        if torch.cuda.is_available():
            if "LOCAL_RANK" in os.environ:
                self.device = int(os.environ["LOCAL_RANK"])
            else:
                self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")
        
    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)
        
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
                torch.matmul(z_flattened, self.embedding.weight.t())
                
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)
        
        z_q = z + (z_q - z).detach()
        
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return loss, z_q, perplexity, min_encodings, min_encoding_indices