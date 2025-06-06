import sys

sys.path.append("..")  

import torch.nn as nn
from models.components import ResnetBlock, Downsample, PreNorm, LinearAttention, Residual


class VQVAEEncoder(nn.Module):
    def __init__(self, in_out, attn=True):
        super().__init__()
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(nn.ModuleList([
            ResnetBlock(dim_in, dim_in),
            ResnetBlock(dim_in, dim_in),
            Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attn else nn.Identity(),
            Downsample(dim_in, dim_out)
        ]))
        self.attn = attn


    def forward(self, x):
        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            x = block2(x)

            x = attn(x)

            x = downsample(x)
       
        return x
