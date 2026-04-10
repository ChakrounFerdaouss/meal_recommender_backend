"""
═══════════════════════════════════════════════════════════════
ÉTAPE 4 — MODÉLISATION
═══════════════════════════════════════════════════════════════

Architecture : CalorieResNet
  → Residual MLP avec LayerNorm + GELU + Dropout

Topologie :
  Input (8)
    → Linear(8→256) + LayerNorm + GELU
    → ResidualBlock × 3  (dim=256)
    → Linear(256→128) + LayerNorm + GELU
    → ResidualBlock × 2  (dim=128)
    → Linear(128→64) + GELU
    → Linear(64→1)

Paramètres : ~270 K  (léger, CPU-friendly)
"""

import torch
import torch.nn as nn

from .config import N_FEATURES


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.ff(x))


class CalorieResNet(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, dropout: float = 0.15):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.res256 = nn.Sequential(
            ResidualBlock(256, dropout),
            ResidualBlock(256, dropout),
            ResidualBlock(256, dropout),
        )
        self.down = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.res128 = nn.Sequential(
            ResidualBlock(128, dropout),
            ResidualBlock(128, dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res256(x)
        x = self.down(x)
        x = self.res128(x)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)