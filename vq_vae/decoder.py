import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components import ResnetBlock, Upsample, PreNorm, LinearAttention, Residual

class VQVAEDecoder(nn.Module):
    def __init__(self, in_out, attn=True):
        super().__init__()

        self.attn = attn
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out , dim_out),
                ResnetBlock(dim_out , dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attn else nn.Identity(),
                Upsample(dim_out, dim_in)
            ]))

    def forward(self, z_q):
        for block1, block2, attn, upsample in self.ups:
            if z_q.shape[-1] == 624:
                z_q = F.pad(z_q, (0, 1, 0, 0))
            
            z_q = block1(z_q)
            z_q = block2(z_q)

            z_q = attn(z_q)

            z_q = upsample(z_q)
        return z_q

