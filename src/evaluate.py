"""The evaluation loop shared by all three runner scripts."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

import policies as P
from metrics import MetricAccumulator


def run_policy(policy: P.Policy, env, cfg: dict, agent, n_sessions: int,
               train_rl_sessions: int = 0, log: Optional[List[Dict]] = None
               ) -> Dict[str, float]:
    """Evaluate one policy for ``n_sessions`` episodes.

    Learning policies are first trained for ``train_rl_sessions`` episodes
    with exploration on, then evaluated with exploration off, so that the
    reported numbers come from a fixed policy rather than from a policy
    still exploring.
    """
    is_rl = isinstance(policy, P.RLDelivery)

    if is_rl and train_rl_sessions:
        policy.training = True
        _rollout(policy, env, cfg, agent, train_rl_sessions, None, None)
        policy.training = False

    acc = MetricAccumulator(cfg)
    _rollout(policy, env, cfg, agent, n_sessions, acc, log)
    return acc.summary()


def _rollout(policy, env, cfg, agent, n_sessions: int,
             acc: Optional[MetricAccumulator], log: Optional[List[Dict]]) -> None:
    steps = cfg["environment"]["steps_per_user"]
    is_rl = isinstance(policy, P.RLDelivery)

    for session in range(n_sessions):
        user = env.reset()
        text = ""
        if getattr(agent, "is_llm", False):
            # The LLM agent needs the utterance itself, not just its features.
            row = env.pool.iloc[env.rng.integers(len(env.pool))]
            text = str(row.text_clean)

        for t in range(steps):
            cands = env.candidates()
            scores = policy.rank(user, cands)
            order = np.argsort(-scores)
            best = int(order[0])
            ad = cands[best]

            if acc is not None:
                truth = env.true_relevance_scores(user, cands)
                acc.add_ranking([float(truth[i]) for i in order])

            state = policy.state(user, float(scores[best])) if is_rl else None
            fatigue_before = user.fatigue
            action = policy.decide(user, ad, float(scores[best]))

            if getattr(agent, "is_llm", False):
                outcome = agent.respond(user, ad, action, text=text)
            else:
                outcome = agent.respond(user, ad, action)

            if is_rl:
                reward = policy.reward(outcome, fatigue_before, user.fatigue)
                next_state = policy.state(user, float(scores[best]))
                policy.observe(state, action, reward, next_state, t == steps - 1)

            if acc is not None:
                acc.add_step(outcome)
            if log is not None:
                log.append({
                    "session": session, "step": t, "policy": policy.name,
                    "agent": getattr(agent, "name", "?"),
                    "action": int(action), "shown": bool(outcome["shown"]),
                    "click": int(outcome["click"]),
                    "negative": int(outcome["negative"]),
                    "p_click": round(float(outcome["p_click"]), 5),
                    "trust": round(float(user.trust), 5),
                    "fatigue": round(float(user.fatigue), 5),
                    "ad_topic": ad.topic,
                    "score": round(float(scores[best]), 5),
                })

        if acc is not None:
            acc.add_session(user)
