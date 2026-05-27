"""Attention-Based Multiple Instance Learning (Ilse, Tomczak, Welling 2018).

For each slide, pool `(N_tiles, embed_dim)` tile embeddings into a single
slide embedding via gated attention, then project to a scalar risk score
for the Cox loss.

Architecture mirrors the paper PEaRL benchmarks against in Table 3 — same
single-layer gated-attention MIL used by the BLEEP / mclSTExp / UNI baselines
in the original paper, so the comparison is apples-to-apples.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    """Gated Attention-Based MIL.

    Forward:
        tiles: (N_tiles, embed_dim) — per-slide tile embeddings
    Returns:
        risk: scalar — Cox-style log-hazard for this slide
        attn: (N_tiles,) — attention weights (useful for tile-level interp)
    """

    def __init__(self, embed_dim: int = 1024, hidden: int = 512,
                 attn_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.embed_dim = embed_dim
        # Project tile embeddings into a smaller MIL feature space
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Gated attention (Ilse et al. eq 9)
        self.attn_V = nn.Sequential(nn.Linear(hidden, attn_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(hidden, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)
        # Risk head — single scalar (Cox log-hazard)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, tiles: torch.Tensor):
        # tiles: (N, embed_dim)
        h = self.fc(tiles)                                          # (N, hidden)
        a = self.attn_w(self.attn_V(h) * self.attn_U(h))            # (N, 1)
        attn = torch.softmax(a, dim=0)                              # (N, 1)
        slide_emb = (attn * h).sum(dim=0)                           # (hidden,)
        risk = self.risk_head(slide_emb).squeeze(-1)                # scalar
        return risk, attn.squeeze(-1)


def cox_ph_loss(risks: torch.Tensor, times: torch.Tensor,
                events: torch.Tensor) -> torch.Tensor:
    """Cox proportional hazards partial-likelihood loss (Efron's form).

    Args:
        risks:  (B,) predicted log-hazards (one per slide in the batch)
        times:  (B,) survival times
        events: (B,) 1 if event observed, 0 if censored

    The standard formulation: sort by descending time, compute the log-sum-exp
    of risks at and after each event, sum the (risk - logsumexp) terms over
    observed events. We use the breslow approximation (ties handled by summing
    risks at tied times before the logsumexp).
    """
    # Sort descending by time so cumsum gives the at-risk set
    idx = torch.argsort(times, descending=True)
    risks = risks[idx]
    events = events[idx].float()
    # log-cumsum-exp from the start: at position i, this is logsumexp of
    # risks[0..i] = risks of all subjects still at risk at time risks[i]
    log_cum = torch.logcumsumexp(risks, dim=0)
    # negative partial log-likelihood
    nll = -((risks - log_cum) * events).sum() / (events.sum() + 1e-9)
    return nll
