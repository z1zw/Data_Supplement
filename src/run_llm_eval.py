"""Evaluation under the simulated user agents -> Table 2 and Fig. 2.

    python src/run_llm_eval.py --agent heuristic
    python src/run_llm_eval.py --agent llm --models gpt-4o,deepseek-v3

--agent heuristic runs the offline user model for every listed agent name
(useful as a free, deterministic dry run of the whole pipeline; the five
'agents' then differ only by seed). --agent llm calls the real models.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model as M
import policies as P
from agents import build_agent
from common import load_config, resolve, set_seed, write_provenance
from env import AdDeliveryEnv
from evaluate import run_policy

DEFAULT_MODELS = ["kimi-2", "deepseek-v3", "gpt-4o", "gemini-1.5-pro", "gpt-5o"]
METRICS = ["CTR", "NDCG", "CRR", "UFI"]


def plot_figure2(df: pd.DataFrame, path) -> None:
    methods = P.LLM_POLICIES
    agents = list(dict.fromkeys(df["user_agent"]))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    x = np.arange(len(agents))
    width = 0.8 / len(methods)

    for ax, metric in zip(axes.ravel(), METRICS):
        for i, method in enumerate(methods):
            sub = df[df["method"] == method].set_index("user_agent")
            vals = [sub.loc[a, metric] if a in sub.index else np.nan for a in agents]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=method)
        ax.set_title(metric + ("  (lower is better)" if metric == "UFI" else ""))
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Performance of delivery strategies under simulated user agents")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--agent", choices=["heuristic", "llm"], default="heuristic")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--sessions", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = resolve(cfg, "data")
    out_dir = resolve(cfg, "results")
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    n_sessions = args.sessions or (cfg["llm"]["n_sessions"] if args.agent == "llm"
                                   else cfg["environment"]["n_users"])

    rows, logs = [], []
    for m_i, key in enumerate(model_keys):
        seed = cfg["seed"] + 100 * (m_i + 1)
        set_seed(seed)
        rng = np.random.default_rng(seed)
        env = AdDeliveryEnv(data_dir, cfg, rng)
        agent = build_agent(args.agent, env, rng, cfg,
                            model_key=key if args.agent == "llm" else None,
                            cache_dir=out_dir / "llm_cache")

        print(f"\n=== user agent: {key} ({args.agent}) seed={seed} ===")
        net = M.train_model(env, cfg, rng, n_sessions=2000, verbose=False)

        for name in P.LLM_POLICIES:
            policy = P.build(name, net, cfg, env)
            log: list = []
            summary = run_policy(policy, env, cfg, agent, n_sessions,
                                 train_rl_sessions=800, log=log)
            rows.append({"user_agent": key, "method": name, "seed": seed,
                         **summary})
            logs.extend(log)
            print(f"  {name:<14} CTR={summary['CTR']:.4f} "
                  f"NDCG={summary['NDCG']:.4f} CRR={summary['CRR']:.3f} "
                  f"UFI={summary['UFI']:.3f}")

        if getattr(agent, "is_llm", False):
            print(f"  [{key}] api calls: {agent.n_called}, cache hits: {agent.n_cached}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table2_llm.csv", index=False)
    pd.DataFrame(logs).to_csv(out_dir / "interaction_log_llm_eval.csv.gz",
                              index=False, compression="gzip")
    plot_figure2(df, out_dir / "fig2.pdf")

    write_provenance(out_dir, "table2_llm", cfg,
                     {"agent_kind": args.agent, "models": model_keys,
                      "sessions": n_sessions})
    print(f"\nWritten to {out_dir / 'table2_llm.csv'} and {out_dir / 'fig2.pdf'}")
    print("Copy these numbers into Table 2 and replace figures/figure2 with fig2.")


if __name__ == "__main__":
    main()
