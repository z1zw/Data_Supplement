"""Delivery policies: the proposed method and every baseline it is compared to.

A policy does two things at each step:
    select() -- rank the candidate advertisements and pick one
    decide() -- choose to show it or to skip

The baselines differ in which of the two they do well:

    CTR-based      ranks by predicted click only, and always shows
    Interest-only  ranks by the interest head only, with a fixed frequency cap
    LLM-Frequency  no ranking (random candidate), fixed frequency cap
    LLM-CTR        ranks by predicted click, fixed frequency cap
    LLM-Interest   ranks by the interest head, fixed frequency cap
    LLM-RL         ranks by the interest head, learned show/skip policy
    Ours           ranks by the joint interest-credibility score (Eq. 5),
                   learned show/skip policy with the fatigue-penalised
                   reward (Eq. 7)

The 'LLM-' prefixed baselines are the same delivery logic evaluated against
LLM user agents rather than the offline user model; the prefix is kept so
the names match Table 2 of the paper.
"""
from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class Policy:
    name = "base"

    def __init__(self, model, cfg: dict, env):
        self.model = model
        self.cfg = cfg
        self.env = env
        self.pc = cfg["policy"]

    def _scores(self, user, ads) -> Tuple[np.ndarray, np.ndarray]:
        mat = np.stack([a.vector(self.env.n_topics) for a in ads])
        return self.model.score(user.vector(), mat)

    def rank(self, user, ads) -> np.ndarray:
        raise NotImplementedError

    def decide(self, user, ad, score: float) -> int:
        raise NotImplementedError

    def observe(self, *args, **kwargs) -> None:      # for learning policies
        pass


class FrequencyCapMixin:
    """Show until the cap is reached inside the rolling window, then skip."""

    def _within_cap(self, user) -> bool:
        window = self.cfg["environment"]["overexposure_window"]
        recent = [s for s in user.recent_impressions if s > user.step - window]
        return len(recent) < self.pc["frequency_cap"]


class CTRBased(Policy):
    name = "CTR-based"

    def rank(self, user, ads):
        s_int, _ = self._scores(user, ads)
        return s_int

    def decide(self, user, ad, score):
        return self.env.SHOW           # always shows: no frequency control


class InterestOnly(FrequencyCapMixin, Policy):
    name = "Interest-only"

    def rank(self, user, ads):
        s_int, _ = self._scores(user, ads)
        return s_int

    def decide(self, user, ad, score):
        return self.env.SHOW if self._within_cap(user) else self.env.SKIP


class RandomFrequency(FrequencyCapMixin, Policy):
    name = "LLM-Frequency"

    def rank(self, user, ads):
        return np.random.rand(len(ads))

    def decide(self, user, ad, score):
        return self.env.SHOW if self._within_cap(user) else self.env.SKIP


class JointScore(Policy):
    """Ranking component of the proposed method: Eq. (5)."""

    def rank(self, user, ads):
        s_int, s_cred = self._scores(user, ads)
        a = self.pc["alpha"]
        return a * s_int + (1.0 - a) * s_cred


# ---------------------------------------------------------------------------
#  Reinforcement-learning show/skip policy
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, d_state: int, d_hidden: int, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class RLDelivery(Policy):
    """Learned show/skip policy over the state of Eq. (6).

    state  = [user vector, joint matching score of the selected ad]
    reward = lambda * engagement - mu * fatigue        (Eq. 7)

    where engagement is the observed click and fatigue is the increase in the
    user's fatigue level plus any negative-feedback event on that step.
    """

    name = "LLM-RL"
    use_joint_score = False

    def __init__(self, model, cfg, env):
        super().__init__(model, cfg, env)
        r = cfg["rl"]
        self.r = r
        self.d_state = env.d_user + 1
        self.q = QNetwork(self.d_state, cfg["representation"]["d_hidden"])
        self.target = QNetwork(self.d_state, cfg["representation"]["d_hidden"])
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=r["lr"])
        self.replay = deque(maxlen=r["replay_size"])
        self.steps_done = 0
        self.training = True

    def rank(self, user, ads):
        s_int, s_cred = self._scores(user, ads)
        if self.use_joint_score:
            a = self.pc["alpha"]
            return a * s_int + (1.0 - a) * s_cred
        return s_int

    def _epsilon(self) -> float:
        r = self.r
        frac = min(1.0, self.steps_done / r["epsilon_decay_steps"])
        return r["epsilon_start"] + frac * (r["epsilon_end"] - r["epsilon_start"])

    def state(self, user, score: float) -> np.ndarray:
        return np.concatenate([user.vector(), [score]]).astype(np.float32)

    def decide(self, user, ad, score):
        s = self.state(user, score)
        self.steps_done += 1
        if self.training and random.random() < self._epsilon():
            return random.choice([self.env.SHOW, self.env.SKIP])
        with torch.no_grad():
            q = self.q(torch.as_tensor(s)[None, :])
        return int(torch.argmax(q, dim=1).item())

    def reward(self, outcome, fatigue_before: float, fatigue_after: float) -> float:
        engagement = float(outcome["click"])
        fatigue = max(0.0, fatigue_after - fatigue_before) + float(outcome["negative"])
        return (self.pc["lambda_engagement"] * engagement
                - self.pc["mu_fatigue"] * fatigue)

    def observe(self, s, a, r, s_next, done):
        if not self.training:
            return
        self.replay.append((s, a, r, s_next, float(done)))
        if len(self.replay) < self.r["warmup_steps"]:
            return
        batch = random.sample(self.replay, self.r["batch_size"])
        S, A, R, S2, D = map(np.asarray, zip(*batch))
        S = torch.as_tensor(S, dtype=torch.float32)
        S2 = torch.as_tensor(S2, dtype=torch.float32)
        A = torch.as_tensor(A, dtype=torch.int64)
        R = torch.as_tensor(R, dtype=torch.float32)
        D = torch.as_tensor(D, dtype=torch.float32)

        q = self.q(S).gather(1, A[:, None]).squeeze(1)
        with torch.no_grad():
            q_next = self.target(S2).max(dim=1).values
            target = R + self.r["gamma"] * (1.0 - D) * q_next
        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.steps_done % self.r["target_sync_every"] == 0:
            self.target.load_state_dict(self.q.state_dict())


class Ours(RLDelivery):
    """The proposed method: joint interest-credibility ranking + RL delivery."""
    name = "Ours"
    use_joint_score = True


class LLMCTR(InterestOnly):
    name = "LLM-CTR"


class LLMInterest(InterestOnly):
    name = "LLM-Interest"


def build(name: str, model, cfg, env) -> Policy:
    table = {
        "CTR-based": CTRBased,
        "Interest-only": InterestOnly,
        "LLM-Frequency": RandomFrequency,
        "LLM-CTR": LLMCTR,
        "LLM-Interest": LLMInterest,
        "LLM-RL": RLDelivery,
        "Ours": Ours,
    }
    if name not in table:
        raise KeyError(f"unknown policy {name!r}; choose from {sorted(table)}")
    return table[name](model, cfg, env)


BENCHMARK_POLICIES: List[str] = ["CTR-based", "Interest-only", "Ours"]
LLM_POLICIES: List[str] = ["LLM-Frequency", "LLM-CTR", "LLM-Interest",
                           "LLM-RL", "Ours"]
