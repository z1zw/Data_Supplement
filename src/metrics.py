"""The four evaluation metrics reported in the paper.

Every formula here is stated in VARIABLES.md; nothing is estimated
implicitly and no metric is smoothed or clipped beyond what is written.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def ctr(n_clicks: int, n_impressions: int) -> float:
    """Click-through rate = clicks / impressions.

    Impressions, not opportunities: a step on which the policy chose to skip
    is not counted in the denominator. This matters, because a policy that
    skips aggressively would otherwise be penalised twice (once here and
    again in the fatigue index).
    """
    return float(n_clicks / n_impressions) if n_impressions else 0.0


def dcg(gains: Sequence[float], k: int) -> float:
    g = np.asarray(gains, dtype=np.float64)[:k]
    discounts = np.log2(np.arange(2, len(g) + 2))
    return float(np.sum(g / discounts))


def ndcg_at_k(ranked_relevance: Sequence[float], k: int) -> float:
    """NDCG@k of one ranking against its own ideal ordering."""
    if len(ranked_relevance) == 0:
        return 0.0
    ideal = sorted(ranked_relevance, reverse=True)
    idcg = dcg(ideal, k)
    return float(dcg(ranked_relevance, k) / idcg) if idcg > 0 else 0.0


def crr(trust_final: Sequence[float], trust_initial: Sequence[float]) -> float:
    """Credibility retention rate = mean over sessions of trust_T / trust_0.

    Values above 1 are possible when a session ends more trusting than it
    began; they are not clipped.
    """
    tf = np.asarray(trust_final, dtype=np.float64)
    t0 = np.asarray(trust_initial, dtype=np.float64)
    if len(tf) == 0:
        return 0.0
    return float(np.mean(tf / np.maximum(t0, 1e-9)))


def interaction_decay(interactions: Sequence[Sequence[int]]) -> float:
    """Mean relative fall in interaction rate from the first to the second
    half of a session. 0 when there is no decay; 1 when interaction stops
    entirely in the second half.
    """
    decays = []
    for seq in interactions:
        if len(seq) < 4:
            continue
        half = len(seq) // 2
        first, second = np.mean(seq[:half]), np.mean(seq[half:])
        if first <= 0:
            continue
        decays.append(max(0.0, (first - second) / first))
    return float(np.mean(decays)) if decays else 0.0


def ufi(user_skip_rate: float, negative_rate: float, decay: float,
        w_skip: float, w_negative: float, w_decay: float) -> float:
    """User fatigue index: a weighted composite of the three fatigue signals.

    UFI = w_skip * user_skip_rate + w_neg * negative_feedback_rate
          + w_decay * interaction_decay

    ``user_skip_rate`` is a *user* behaviour -- the share of shown
    advertisements the user scrolled past without clicking. It is NOT the
    rate at which the policy chose to withhold an advertisement. The
    distinction matters: counting policy-side withholding here would punish
    a policy for protecting the user from over-exposure, which is the exact
    behaviour the fatigue index is supposed to reward.

    Weights are set in config.yaml and sum to 1, so UFI lies in [0, 1] and
    lower is better.
    """
    return float(w_skip * user_skip_rate + w_negative * negative_rate
                 + w_decay * decay)


class MetricAccumulator:
    """Collects the raw counts needed for all four metrics across sessions."""

    def __init__(self, cfg: dict):
        self.m = cfg["metrics"]
        self.impressions = 0
        self.clicks = 0
        self.skips = 0
        self.steps = 0
        self.negatives = 0
        self.ndcgs: List[float] = []
        self.trust_final: List[float] = []
        self.trust_initial: List[float] = []
        self.interactions: List[List[int]] = []

    def add_step(self, outcome: Dict) -> None:
        self.steps += 1
        if outcome["shown"]:
            self.impressions += 1
            self.clicks += outcome["click"]
        else:
            self.skips += 1
        self.negatives += outcome["negative"]

    def add_ranking(self, ranked_relevance: Sequence[float]) -> None:
        self.ndcgs.append(ndcg_at_k(ranked_relevance, self.m["ndcg_k"]))

    def add_session(self, user) -> None:
        self.trust_final.append(user.trust)
        self.trust_initial.append(user.trust_initial)
        self.interactions.append(list(user.interactions))

    def summary(self) -> Dict[str, float]:
        # User skip rate: shown but not clicked. See ufi() for why this is
        # not the policy's own withhold rate.
        user_skip_rate = ((self.impressions - self.clicks) / self.impressions
                          if self.impressions else 0.0)
        withhold_rate = self.skips / self.steps if self.steps else 0.0
        neg_rate = self.negatives / self.impressions if self.impressions else 0.0
        decay = interaction_decay(self.interactions)
        return {
            "CTR": ctr(self.clicks, self.impressions),
            "NDCG": float(np.mean(self.ndcgs)) if self.ndcgs else 0.0,
            "CRR": crr(self.trust_final, self.trust_initial),
            "UFI": ufi(user_skip_rate, neg_rate, decay,
                       self.m["ufi_w_skip"], self.m["ufi_w_negative"],
                       self.m["ufi_w_decay"]),
            "_impressions": self.impressions,
            "_clicks": self.clicks,
            "_user_skip_rate": user_skip_rate,
            "_policy_withhold_rate": withhold_rate,
            "_negative_rate": neg_rate,
            "_interaction_decay": decay,
            "_sessions": len(self.trust_final),
        }
