import torch
import torch.nn as nn
import torch.nn.functional as F


class VQVAEDecoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, kernel_size=(3, 6), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 6), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 6), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3)),
        )

    def forward(self, z_q):
        x = self.block1(z_q)
        x = self.block2(x)
        x = self.block3(x)
        if x.shape[-1] % 2 != 1:
            x = F.pad(x, (0, 1, 0, 0))
        x = self.block4(x)
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0))
        x = self.block5(x)
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0))
        return x

