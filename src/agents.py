"""Simulated user agents.

Two kinds:

    HeuristicAgent -- the offline user model. Responses are drawn from the
                      environment's generative click model. Deterministic
                      given the seed, free to run, and what a reviewer can
                      execute without API keys.

    LLMAgent       -- a large language model acting as the user. It is shown
                      the sentiment text, the advertisement description and a
                      summary of the interaction so far, and returns a
                      judgement of interest, credibility and whether it would
                      click. No web search and no tool use, exactly as stated
                      in the Methods section.

Every LLM response is cached on disk under results/llm_cache/, keyed by a
hash of the exact prompt. The cache is the audit trail for the LLM
conditions and should be shipped with the submission.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from common import ROOT

PROMPT_DIR = ROOT / "prompts"


class HeuristicAgent:
    """Offline user model: responses come from the environment itself."""

    name = "heuristic"
    is_llm = False

    def __init__(self, env, rng: np.random.Generator):
        self.env = env
        self.rng = rng

    def respond(self, user, ad, action: int) -> Dict:
        return self.env.step(user, ad, action)


class LLMAgent:
    """A large language model acting as the simulated user.

    The model's judgement replaces the environment's click draw. Trust and
    fatigue still evolve through the environment's dynamics, so that the two
    agent types remain comparable on CRR and UFI.
    """

    is_llm = True

    def __init__(self, env, rng: np.random.Generator, cfg: dict,
                 model_key: str, cache_dir: Path):
        self.env = env
        self.rng = rng
        self.cfg = cfg
        self.name = model_key
        spec = cfg["llm"]["endpoints"][model_key]
        self.provider = spec["provider"]
        self.model = spec["model"]
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = (PROMPT_DIR / "user_agent_system.txt").read_text(
            encoding="utf-8")
        self.turn_template = (PROMPT_DIR / "user_agent_turn.txt").read_text(
            encoding="utf-8")
        self.n_cached = 0
        self.n_called = 0

    # -- prompt construction ----------------------------------------------
    def build_prompt(self, user, ad, text: str) -> str:
        history = user.interactions[-5:]
        summary = (f"{sum(history)} clicks out of the last {len(history)} "
                   f"advertisements") if history else "no advertisements seen yet"
        return self.turn_template.format(
            sentiment_text=text,
            valence="positive" if user.sentiment >= 0 else "negative",
            arousal=f"{user.arousal:.2f}",
            ad_topic=ad.topic,
            ad_trust=f"{ad.trust_prior:.2f}",
            ad_intrusiveness=f"{ad.intrusiveness:.2f}",
            history_summary=summary,
            n_seen=len(user.interactions),
        )

    # -- provider calls ----------------------------------------------------
    def _cache_path(self, prompt: str) -> Path:
        key = hashlib.sha256(
            f"{self.provider}|{self.model}|{prompt}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{self.name}_{key}.json"

    def _call(self, prompt: str) -> str:
        import requests

        llm = self.cfg["llm"]
        if self.provider in ("openai", "deepseek", "moonshot"):
            base, env_key = {
                "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
                "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
                "moonshot": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
            }[self.provider]
            key = os.environ.get(env_key)
            if not key:
                raise RuntimeError(f"{env_key} is not set")
            resp = requests.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": llm["temperature"],
                    "max_tokens": llm["max_tokens"],
                },
                timeout=llm["timeout_s"],
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        if self.provider == "gemini":
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.model}:generateContent?key={key}",
                json={
                    "system_instruction": {"parts": [{"text": self.system_prompt}]},
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": llm["temperature"],
                        "maxOutputTokens": llm["max_tokens"],
                    },
                },
                timeout=llm["timeout_s"],
            )
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

        raise ValueError(f"unknown provider {self.provider!r}")

    def query(self, prompt: str) -> Dict:
        path = self._cache_path(prompt)
        if path.exists():
            self.n_cached += 1
            return json.loads(path.read_text(encoding="utf-8"))["parsed"]

        raw = self._call(prompt)
        self.n_called += 1
        parsed = self.parse(raw)
        path.write_text(
            json.dumps({"prompt": prompt, "raw": raw, "parsed": parsed},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        return parsed

    @staticmethod
    def parse(raw: str) -> Dict:
        """Extract the JSON object the agent was asked to return.

        A response that cannot be parsed is recorded as a non-click with
        neutral credibility rather than being silently dropped, so that
        parse failures show up in the results instead of biasing them.
        """
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return {"click": 0, "interest": 0.0, "credibility": 0.5,
                    "parse_failed": True}
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"click": 0, "interest": 0.0, "credibility": 0.5,
                    "parse_failed": True}
        return {
            "click": int(bool(obj.get("click", 0))),
            "interest": float(obj.get("interest", 0.0)),
            "credibility": float(obj.get("credibility", 0.5)),
            "parse_failed": False,
        }

    # -- environment interface --------------------------------------------
    def respond(self, user, ad, action: int, text: str = "") -> Dict:
        if action != self.env.SHOW:
            return self.env.step(user, ad, action)

        judgement = self.query(self.build_prompt(user, ad, text))
        # The agent decides the click; the environment applies the trust and
        # fatigue dynamics so that all conditions share the same accounting.
        out = self.env.step(user, ad, action)
        delta = judgement["click"] - out["click"]
        out["click"] = judgement["click"]
        out["llm_interest"] = judgement["interest"]
        out["llm_credibility"] = judgement["credibility"]
        out["parse_failed"] = judgement["parse_failed"]
        if delta != 0:
            # Keep trust consistent with the agent's decision.
            e = self.env.env_cfg
            user.trust = float(np.clip(
                user.trust + delta * e["trust_gain_on_click"],
                e["trust_floor"], e["trust_ceiling"]))
            out["trust_after"] = user.trust
            if user.interactions:
                user.interactions[-1] = judgement["click"]
        return out


def build_agent(kind: str, env, rng, cfg, model_key: Optional[str] = None,
                cache_dir: Optional[Path] = None):
    if kind == "heuristic":
        return HeuristicAgent(env, rng)
    if kind == "llm":
        if not model_key:
            raise ValueError("model_key is required for the llm agent")
        return LLMAgent(env, rng, cfg, model_key,
                        cache_dir or (ROOT / "results" / "llm_cache"))
    raise ValueError(f"unknown agent kind {kind!r}")
