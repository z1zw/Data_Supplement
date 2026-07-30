"""Deep Sentiment Network.

A shared trunk over the concatenated user and advertisement representations,
with two separate output heads: one predicting instantaneous interest
(Eq. 3) and one predicting perceived credibility (Eq. 4). The heads share
the trunk so that the two signals are learned from a common representation,
but they are predicted separately so that neither masks the other.

Training targets:
    interest head    -> the observed click on that impression
    credibility head -> whether the impression preserved the user's trust
                        (1 if trust did not fall, 0 if it fell)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class DeepSentimentNetwork(nn.Module):
    def __init__(self, d_user: int, d_ad: int, d_hidden: int, dropout: float):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_user + d_ad, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )
        self.interest_head = nn.Linear(d_hidden, 1)
        self.credibility_head = nn.Linear(d_hidden, 1)

    def forward(self, u: torch.Tensor, a: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(torch.cat([u, a], dim=-1))
        return (torch.sigmoid(self.interest_head(z)).squeeze(-1),
                torch.sigmoid(self.credibility_head(z)).squeeze(-1))

    @torch.no_grad()
    def score(self, u: np.ndarray, ads: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
        """Score one user against a batch of candidate ads."""
        self.eval()
        ut = torch.as_tensor(np.repeat(u[None, :], len(ads), axis=0))
        at = torch.as_tensor(ads)
        s_int, s_cred = self.forward(ut, at)
        return s_int.numpy(), s_cred.numpy()


def collect_training_data(env, cfg, rng, n_sessions: int
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Behaviour-policy rollouts used to fit the network.

    The exploration policy shows a uniformly random candidate at every step,
    so the training set is unbiased with respect to the scoring functions
    that are being learned. This is deliberate: training on the proposed
    policy's own impressions would make the interest head look better than
    it is.
    """
    steps = cfg["environment"]["steps_per_user"]
    U, A, Y_int, Y_cred = [], [], [], []
    for _ in range(n_sessions):
        user = env.reset()
        for _ in range(steps):
            cands = env.candidates()
            ad = cands[rng.integers(len(cands))]
            u_vec = user.vector().copy()
            a_vec = ad.vector(env.n_topics)
            out = env.step(user, ad, env.SHOW)
            U.append(u_vec)
            A.append(a_vec)
            Y_int.append(float(out["click"]))
            Y_cred.append(float(out["trust_after"] >= out["trust_before"]))
    return (np.asarray(U, dtype=np.float32), np.asarray(A, dtype=np.float32),
            np.asarray(Y_int, dtype=np.float32), np.asarray(Y_cred, dtype=np.float32))


def train_model(env, cfg: dict, rng, n_sessions: int = 3000,
                verbose: bool = True) -> DeepSentimentNetwork:
    mc = cfg["model"]
    U, A, Yi, Yc = collect_training_data(env, cfg, rng, n_sessions)
    model = DeepSentimentNetwork(env.d_user, env.d_ad,
                                 cfg["representation"]["d_hidden"], mc["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=mc["lr"],
                           weight_decay=mc["weight_decay"])
    loss_fn = nn.BCELoss()

    Ut, At = torch.as_tensor(U), torch.as_tensor(A)
    Yit, Yct = torch.as_tensor(Yi), torch.as_tensor(Yc)
    n, bs = len(U), mc["batch_size"]

    for epoch in range(mc["epochs"]):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            p_int, p_cred = model(Ut[idx], At[idx])
            loss = loss_fn(p_int, Yit[idx]) + loss_fn(p_cred, Yct[idx])
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        if verbose:
            print(f"  epoch {epoch + 1}/{mc['epochs']}  loss={total / n:.4f}")
    return model


def save(model: DeepSentimentNetwork, path: Path) -> None:
    torch.save(model.state_dict(), path)


def load(path: Path, d_user: int, d_ad: int, cfg: dict) -> DeepSentimentNetwork:
    model = DeepSentimentNetwork(d_user, d_ad, cfg["representation"]["d_hidden"],
                                 cfg["model"]["dropout"])
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
