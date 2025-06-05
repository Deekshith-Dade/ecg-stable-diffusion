import torch
import torch.nn as nn

class VQVAEEncoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 6), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 6), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(128, embedding_dim, kernel_size=(3, 6), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
