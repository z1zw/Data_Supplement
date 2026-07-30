"""Benchmark simulation -> Table 1.

Compares the proposed method against the CTR-based and interest-only
baselines in the mixed user agent environment.

Usage:
    python src/run_benchmark.py [--sessions 2000] [--repeats 5]

With --repeats > 1 the whole experiment is repeated under different seeds
and the output reports mean and standard deviation across repeats. Report
those, not a single run: the journal's statistical guidelines ask for a
measure of centre and a measure of variability for every reported quantity.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import model as M
import policies as P
from common import ROOT, load_config, resolve, set_seed, write_provenance
from env import AdDeliveryEnv
from evaluate import run_policy
from agents import build_agent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--sessions", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--train-sessions", type=int, default=3000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = resolve(cfg, "data")
    out_dir = resolve(cfg, "results")
    n_sessions = args.sessions or cfg["environment"]["n_users"]

    rows = []
    for repeat in range(args.repeats):
        seed = cfg["seed"] + repeat
        set_seed(seed)
        rng = np.random.default_rng(seed)
        env = AdDeliveryEnv(data_dir, cfg, rng)
        agent = build_agent("heuristic", env, rng, cfg)

        print(f"\n[repeat {repeat + 1}/{args.repeats}] seed={seed}")
        print("  training the Deep Sentiment Network ...")
        net = M.train_model(env, cfg, rng, n_sessions=args.train_sessions)

        for name in P.BENCHMARK_POLICIES:
            policy = P.build(name, net, cfg, env)
            summary = run_policy(policy, env, cfg, agent, n_sessions,
                                 train_rl_sessions=1000)
            rows.append({"repeat": repeat, "seed": seed, "method": name, **summary})
            print(f"  {name:<14} CTR={summary['CTR']:.4f} "
                  f"NDCG={summary['NDCG']:.4f} CRR={summary['CRR']:.3f} "
                  f"UFI={summary['UFI']:.3f}")

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "table1_raw_runs.csv", index=False)

    agg = raw.groupby("method")[["CTR", "NDCG", "CRR", "UFI"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reindex(P.BENCHMARK_POLICIES).reset_index()
    agg.to_csv(out_dir / "table1_benchmark.csv", index=False)

    print("\n=== Table 1 (mean +/- s.d. over "
          f"{args.repeats} repeats, n={n_sessions} sessions each) ===")
    for _, r in agg.iterrows():
        print(f"{r['method']:<14} "
              f"CTR {r['CTR_mean']:.4f}+/-{r['CTR_std']:.4f}  "
              f"NDCG {r['NDCG_mean']:.4f}+/-{r['NDCG_std']:.4f}  "
              f"CRR {r['CRR_mean']:.3f}+/-{r['CRR_std']:.3f}  "
              f"UFI {r['UFI_mean']:.3f}+/-{r['UFI_std']:.3f}")

    write_provenance(out_dir, "table1_benchmark", cfg,
                     {"sessions": n_sessions, "repeats": args.repeats,
                      "train_sessions": args.train_sessions})
    print(f"\nWritten to {out_dir / 'table1_benchmark.csv'}")
    print("Copy these numbers into Table 1 of the manuscript, including the "
          "standard deviations.")


if __name__ == "__main__":
    main()
