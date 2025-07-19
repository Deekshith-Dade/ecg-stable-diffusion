import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from ..vqvae_base import QuantizeOutput


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 /
                                            self.num_embeddings, 1.0 / self.num_embeddings)
        if torch.cuda.is_available():
            if "LOCAL_RANK" in os.environ:
                self.device = int(os.environ["LOCAL_RANK"])
            else:
                self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

    def forward(self, z) -> QuantizeOutput:
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.num_embeddings).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())

        z_q = z + (z_q - z).detach()

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return QuantizeOutput(z_q, perplexity, {'commitment_loss': commitment_loss, 'codebook_loss': codebook_loss}, min_encodings, min_encoding_indices)


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 /
                                            self.num_embeddings, 1.0 / self.num_embeddings)

        self.register_buffer("ema_cluster_size",
                             torch.zeros(self.num_embeddings))
        self.ema_w = nn.Parameter(torch.Tensor(
            self.num_embeddings, self.embedding_dim))
        self.ema_w.data.normal_()

        if torch.cuda.is_available():
            if "LOCAL_RANK" in os.environ:
                self.device = int(os.environ["LOCAL_RANK"])
            else:
                self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

    def forward(self, z) -> QuantizeOutput:
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.num_embeddings).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                (1 - self.decay) * torch.sum(min_encodings, 0)

            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            dw = torch.matmul(min_encodings.t(), z_flattened)
            self.ema_w = nn.Parameter(
                self.ema_w * self.decay + (1 - self.decay) * dw)

            self.embedding.weight = nn.Parameter(
                self.ema_w / self.ema_cluster_size.unsqueeze(1))

        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = torch.tensor(0.0)

        z_q = z + (z_q - z).detach()

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return QuantizeOutput(z_q, perplexity, {'commitment_loss': commitment_loss, 'codebook_loss': codebook_loss}, min_encodings, min_encoding_indices)
