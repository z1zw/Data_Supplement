"""The advertising delivery simulator.

Implements the environment described in the Methods section: a user state
that carries sentiment, arousal, a latent interest vector, an evolving trust
level and an accumulating fatigue level; an advertisement inventory built
from Avazu; and a generative click model.

Every coefficient lives in config.yaml. The click model, the trust dynamics
and the fatigue dynamics are explicit modelling assumptions, documented in
VARIABLES.md -- they are the simulator, not empirical findings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Ad:
    index: int
    topic: int
    trust_prior: float
    intrusiveness: float
    hashed: np.ndarray

    def vector(self, n_topics: int) -> np.ndarray:
        onehot = np.zeros(n_topics, dtype=np.float32)
        onehot[self.topic] = 1.0
        return np.concatenate(
            [onehot, [self.trust_prior, self.intrusiveness], self.hashed]
        ).astype(np.float32)


@dataclass
class UserState:
    sentiment: float                 # [-1, 1]
    arousal: float                   # [0, 1]
    interest: np.ndarray             # simplex over n_topics
    trust: float                     # [0, 1]
    fatigue: float                   # [0, 1]
    trust_initial: float
    step: int = 0
    recent_impressions: List[int] = field(default_factory=list)
    interactions: List[int] = field(default_factory=list)
    skips: int = 0
    negatives: int = 0

    def vector(self) -> np.ndarray:
        return np.concatenate(
            [[self.sentiment, self.arousal], self.interest,
             [self.trust, self.fatigue]]
        ).astype(np.float32)


class AdDeliveryEnv:
    """Sequential ad delivery environment.

    One episode = one simulated user session of ``steps_per_user`` steps.
    At each step the policy receives ``candidates_per_step`` advertisements,
    picks one, and decides whether to show it.
    """

    SHOW, SKIP = 1, 0

    def __init__(self, data_dir: Path, cfg: dict, rng: np.random.Generator,
                 drift_prob: float | None = None):
        self.cfg = cfg
        self.env_cfg = cfg["environment"]
        self.n_topics = cfg["preprocess"]["n_topics"]
        self.rng = rng
        self.drift_prob = (self.env_cfg["interest_drift_prob"]
                           if drift_prob is None else drift_prob)

        pool = pd.read_csv(data_dir / "sentiment_pool.csv.gz")
        ads = pd.read_csv(data_dir / "ad_pool.csv.gz")
        self.pool = pool.reset_index(drop=True)
        hash_cols = [c for c in ads.columns if c.startswith("h")]
        self.ads = [
            Ad(index=i,
               topic=int(row.topic),
               trust_prior=float(row.trust_prior),
               intrusiveness=float(row.intrusiveness),
               hashed=row[hash_cols].to_numpy(dtype=np.float32))
            for i, row in ads.iterrows()
        ]
        self.d_user = 2 + self.n_topics + 2
        self.d_ad = self.n_topics + 2 + len(hash_cols)

    # -- user construction -------------------------------------------------
    def _sample_interest(self, topic: int) -> np.ndarray:
        """Interest vector concentrated on the utterance's topic."""
        alpha = np.full(self.n_topics, 0.4)
        alpha[topic] = 3.0
        return self.rng.dirichlet(alpha).astype(np.float32)

    def reset(self) -> UserState:
        row = self.pool.iloc[self.rng.integers(len(self.pool))]
        trust0 = self.env_cfg["trust_init"]
        return UserState(
            sentiment=float(row.sentiment),
            arousal=float(row.arousal),
            interest=self._sample_interest(int(row.topic)),
            trust=trust0,
            fatigue=0.0,
            trust_initial=trust0,
        )

    def candidates(self) -> List[Ad]:
        n = self.env_cfg["candidates_per_step"]
        idx = self.rng.integers(len(self.ads), size=n)
        return [self.ads[i] for i in idx]

    # -- ground truth ------------------------------------------------------
    def relevance(self, user: UserState, ad: Ad) -> float:
        """Latent interest-context match: the user's interest in the ad's topic."""
        return float(user.interest[ad.topic])

    def mood_congruence(self, user: UserState, ad: Ad) -> float:
        """Alignment between the user's affective state and the ad's tone.

        Positive-valence users are assumed more receptive to high-trust
        inventory; negative-valence users are assumed more sensitive to
        intrusiveness. Documented assumption -- see VARIABLES.md.
        """
        if user.sentiment >= 0:
            return ad.trust_prior
        return 1.0 - ad.intrusiveness

    def click_probability(self, user: UserState, ad: Ad) -> float:
        e = self.env_cfg
        logit = (e["click_bias"]
                 + e["w_relevance"] * self.relevance(user, ad)
                 + e["w_trust"] * user.trust
                 + e["w_mood"] * self.mood_congruence(user, ad)
                 - e["w_fatigue"] * user.fatigue)
        return float(sigmoid(logit))

    def true_relevance_scores(self, user: UserState, ads: List[Ad]) -> np.ndarray:
        """Graded relevance used as the NDCG ground truth."""
        return np.array([self.relevance(user, a) * (0.5 + 0.5 * a.trust_prior)
                         for a in ads], dtype=np.float32)

    # -- transition --------------------------------------------------------
    def step(self, user: UserState, ad: Ad | None, action: int) -> Dict:
        """Advance one time step. ``action`` is SHOW or SKIP."""
        e = self.env_cfg
        out = {"shown": False, "click": 0, "negative": 0, "skip": 0,
               "trust_before": user.trust, "p_click": 0.0}

        if action == self.SHOW and ad is not None:
            out["shown"] = True
            p = self.click_probability(user, ad)
            out["p_click"] = p
            click = int(self.rng.random() < p)
            out["click"] = click

            user.recent_impressions.append(user.step)
            user.interactions.append(click)
            user.fatigue = min(1.0, user.fatigue + e["fatigue_per_impression"])

            if click:
                user.trust += e["trust_gain_on_click"]
            elif self.relevance(user, ad) < e["mismatch_threshold"]:
                user.trust -= e["trust_loss_on_mismatch"]
                out["negative"] = 1
                user.negatives += 1

            window = [s for s in user.recent_impressions
                      if s > user.step - e["overexposure_window"]]
            if len(window) > e["overexposure_limit"]:
                user.trust -= e["trust_loss_on_overexposure"]
                out["negative"] = 1
                user.negatives += 1
        else:
            out["skip"] = 1
            user.skips += 1

        user.fatigue = max(0.0, user.fatigue - e["fatigue_decay_per_step"])
        user.trust = float(np.clip(user.trust, e["trust_floor"], e["trust_ceiling"]))

        # Short-term interest drift.
        if self.rng.random() < self.drift_prob:
            new_topic = int(self.rng.integers(self.n_topics))
            user.interest = self._sample_interest(new_topic)

        user.step += 1
        out["trust_after"] = user.trust
        return out
