"""Robustness under increasing interest drift -> Fig. 3.

Sweeps the interest drift probability and records how each policy degrades.

    python src/run_robustness.py [--repeats 3]
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

POLICIES = ["CTR-based", "Interest-only", "LLM-RL", "Ours"]


def plot_figure3(agg: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for metric, ax in zip(["CTR", "CRR"], axes):
        for name in POLICIES:
            sub = agg[agg["method"] == name].sort_values("drift_prob")
            ax.errorbar(sub["drift_prob"], sub[f"{metric}_mean"],
                        yerr=sub[f"{metric}_std"], marker="o", capsize=3,
                        label=name)
        ax.set_xlabel("interest drift probability per step")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Robustness under increasing user interest shift frequency")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = resolve(cfg, "data")
    out_dir = resolve(cfg, "results")
    drifts = cfg["robustness"]["drift_probs"]

    rows = []
    for repeat in range(args.repeats):
        for p_drift in drifts:
            seed = cfg["seed"] + 1000 * repeat + int(p_drift * 100)
            set_seed(seed)
            rng = np.random.default_rng(seed)
            env = AdDeliveryEnv(data_dir, cfg, rng, drift_prob=p_drift)
            agent = build_agent("heuristic", env, rng, cfg)
            net = M.train_model(env, cfg, rng, n_sessions=2000, verbose=False)

            for name in POLICIES:
                policy = P.build(name, net, cfg, env)
                summary = run_policy(policy, env, cfg, agent, args.sessions,
                                     train_rl_sessions=800)
                rows.append({"repeat": repeat, "drift_prob": p_drift,
                             "method": name, "seed": seed, **summary})
            print(f"[repeat {repeat + 1}] drift={p_drift:.1f} done")

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "robustness_raw_runs.csv", index=False)

    agg = (raw.groupby(["method", "drift_prob"])[["CTR", "NDCG", "CRR", "UFI"]]
              .agg(["mean", "std"]))
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_dir / "robustness.csv", index=False)
    plot_figure3(agg, out_dir / "fig3.pdf")

    write_provenance(out_dir, "robustness", cfg,
                     {"repeats": args.repeats, "sessions": args.sessions,
                      "drift_probs": drifts})
    print(f"\nWritten to {out_dir / 'robustness.csv'} and {out_dir / 'fig3.pdf'}")
    print("Replace figures/figure3 with fig3, and report the degradation "
          "slope you actually observe.")


if __name__ == "__main__":
    main()
