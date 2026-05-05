"""
model.py
────────
Arquitetura MLP para classificação binária spam / ham.
"""

import torch.nn as nn

from config import FEATURE_DIM


class SpamMLP(nn.Module):
    """MLP de duas camadas ocultas para detecção de spam."""

    def __init__(self, input_dim: int = FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)
