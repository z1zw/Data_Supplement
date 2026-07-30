"""Build the cleaned datasets used by every experiment.

Reads the three public source datasets from ``raw/`` and writes cleaned,
documented tables to ``data/``:

    data/sentiment_pool.csv.gz   user sentiment texts  (Sentiment140 + Weibo)
    data/ad_pool.csv.gz          advertisement inventory (Avazu)
    data/manifest.json            record counts before/after every filter

Every transformation applied here is described in VARIABLES.md.

Usage:
    python src/preprocess.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, load_config, resolve, set_seed, stable_hash

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")
REPEAT_RE = re.compile(r"(.)\1{2,}")           # "sooooo" -> "soo"
EXCLAIM_RE = re.compile(r"[!?！？]")


def clean_text(text: str) -> str:
    """Denoising + normalisation. Applied identically to both languages."""
    text = URL_RE.sub(" ", str(text))
    text = MENTION_RE.sub(" ", text)
    text = REPEAT_RE.sub(r"\1\1", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def arousal_proxy(raw_text: str) -> float:
    """Arousal proxy in [0, 1].

    Derived from surface intensity markers only (exclamation/question marks,
    proportion of upper-case characters, text length). This is a documented
    proxy, not a validated psychometric measure -- see VARIABLES.md.
    """
    raw = str(raw_text)
    if not raw:
        return 0.0
    marks = min(len(EXCLAIM_RE.findall(raw)), 5) / 5.0
    letters = [c for c in raw if c.isalpha()]
    caps = (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0
    length = min(len(raw), 280) / 280.0
    return float(np.clip(0.5 * marks + 0.3 * caps + 0.2 * length, 0.0, 1.0))


def _balanced_sample(df: pd.DataFrame, label_col: str, n_rows: int,
                     seed: int) -> pd.DataFrame:
    """Take an equal number of rows per class, sampled from the whole file.

    Both source files are stored sorted by label -- the first 200,000 rows of
    Sentiment140 are entirely negative. Reading with ``nrows=`` would
    therefore yield a single-class corpus. The whole file is read and then
    sampled, so the class balance is controlled rather than accidental.
    """
    per_class = max(1, n_rows // df[label_col].nunique())
    parts = [
        g.sample(min(len(g), per_class), random_state=seed)
        for _, g in df.groupby(label_col, sort=True)
    ]
    return (pd.concat(parts)
              .sample(frac=1.0, random_state=seed)
              .reset_index(drop=True))


def load_sentiment140(path: Path, n_rows: int, seed: int) -> pd.DataFrame:
    cols = ["target", "id", "date", "flag", "user", "text"]
    df = pd.read_csv(path, encoding="latin-1", names=cols)
    # Coding scheme: Sentiment140 target 0 = negative, 4 = positive.
    df = df[df["target"].isin([0, 4])].copy()
    df = _balanced_sample(df, "target", n_rows, seed)
    df["sentiment"] = np.where(df["target"] == 4, 1.0, -1.0)
    df["language"] = "en"
    return df[["text", "sentiment", "language"]]


def load_weibo(path: Path, n_rows: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Coding scheme: Weibo_Senti_100k label 1 = positive, 0 = negative.
    label_col = "label" if "label" in df.columns else df.columns[0]
    text_col = "review" if "review" in df.columns else df.columns[1]
    df = df[df[label_col].isin([0, 1])].copy()
    df = _balanced_sample(df, label_col, n_rows, seed)
    df["sentiment"] = np.where(df[label_col] == 1, 1.0, -1.0)
    df["language"] = "zh"
    df = df.rename(columns={text_col: "text"})
    return df[["text", "sentiment", "language"]]


def build_sentiment_pool(cfg: dict, raw: Path, manifest: dict) -> pd.DataFrame:
    pp = cfg["preprocess"]
    frames = []

    s140 = raw / "training.1600000.processed.noemoticon.csv"
    if s140.exists():
        df = load_sentiment140(s140, pp["sentiment140_rows"], cfg["seed"])
        manifest["sentiment140_sampled"] = len(df)
        manifest["sentiment140_class_balance"] = (
            df["sentiment"].value_counts().to_dict())
        frames.append(df)
    else:
        raise FileNotFoundError(
            f"{s140} not found. See README section 3 for the download link.")

    weibo = raw / "weibo_senti_100k.csv"
    if weibo.exists():
        df = load_weibo(weibo, pp["weibo_rows"], cfg["seed"])
        manifest["weibo_sampled"] = len(df)
        manifest["weibo_class_balance"] = (
            df["sentiment"].value_counts().to_dict())
        frames.append(df)
    else:
        raise FileNotFoundError(
            f"{weibo} not found. See README section 3 for the download link.")

    pool = pd.concat(frames, ignore_index=True)
    manifest["sentiment_pool_before_filter"] = len(pool)

    pool["arousal"] = pool["text"].map(arousal_proxy)
    pool["text_clean"] = pool["text"].map(clean_text)
    pool["text_clean"] = pool["text_clean"].str.slice(0, pp["max_text_chars"])

    keep = pool["text_clean"].str.len() >= pp["min_text_chars"]
    manifest["dropped_too_short"] = int((~keep).sum())
    pool = pool[keep].copy()

    before_dedup = len(pool)
    pool = pool.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)
    manifest["dropped_duplicates"] = before_dedup - len(pool)

    # Latent interest topic of the utterance: deterministic hash bucket of the
    # cleaned text. Stands in for an unsupervised topic assignment.
    k = pp["n_topics"]
    pool["topic"] = [stable_hash(t, k) for t in pool["text_clean"]]

    manifest["sentiment_pool_final"] = len(pool)
    return pool[["text_clean", "sentiment", "arousal", "topic", "language"]]


# Fields used to define an "advertisement". The Kaggle original names them
# explicitly; some public mirrors ship the same columns anonymised as
# feat_1..feat_22. Both are supported -- see resolve_avazu_schema().
AVAZU_CATEGORICAL = [
    "C1", "banner_pos", "site_category", "app_category",
    "device_type", "device_conn_type", "C15", "C16", "C18",
]

# Position, within the categorical list, of the field used for the
# intrusiveness proxy (banner_pos in the original schema).
AVAZU_INTRUSIVENESS_IDX = 1


def resolve_avazu_schema(src: Path, manifest: dict) -> tuple:
    """Work out which Avazu variant this file is, and which columns to use.

    Returns (label_column, categorical_columns, intrusiveness_column).

    Raises if the file is a mirror whose columns are anonymised, because the
    intrusiveness proxy needs to know which column is banner_pos and that
    mapping is not published for those mirrors. Guessing it would put an
    undocumented assumption into the results.
    """
    header = pd.read_csv(src, nrows=0).columns.tolist()

    if "click" in header and set(AVAZU_CATEGORICAL).issubset(header):
        manifest["avazu_schema"] = "kaggle_original"
        return "click", AVAZU_CATEGORICAL, "banner_pos"

    if "label" in header and set(AVAZU_CATEGORICAL).issubset(header):
        # Some mirrors rename only the target column.
        manifest["avazu_schema"] = "mirror_named_fields"
        return "label", AVAZU_CATEGORICAL, "banner_pos"

    anonymised = [c for c in header if c.startswith("feat_")]
    if anonymised:
        raise ValueError(
            f"{src.name} uses anonymised column names ({anonymised[:3]}...). "
            "The mapping from feat_N to the original Avazu fields is not "
            "published for this mirror, so banner_pos cannot be identified "
            "and the intrusiveness variable cannot be constructed. Use the "
            "Kaggle original or the Avazu_x4 mirror, both of which keep the "
            "original field names. See README section 3.")

    raise ValueError(
        f"Unrecognised Avazu schema in {src.name}. Columns found: {header[:8]}")


def build_ad_pool(cfg: dict, raw: Path, manifest: dict) -> pd.DataFrame:
    """Advertisement inventory derived from Avazu, streamed in chunks."""
    pp, rep = cfg["preprocess"], cfg["representation"]
    src = None
    for candidate in ("train.csv", "train.gz", "train.csv.gz"):
        if (raw / candidate).exists():
            src = raw / candidate
            break
    if src is None:
        raise FileNotFoundError(
            f"No Avazu train file found in {raw}. See README section 3.")

    label_col, categorical, intrusive_col = resolve_avazu_schema(src, manifest)

    usecols = [label_col] + categorical
    chunks, seen = [], 0
    for chunk in pd.read_csv(src, usecols=usecols, chunksize=500_000,
                             dtype=str, compression="infer"):
        chunk[label_col] = chunk[label_col].astype(int)
        chunks.append(chunk)
        seen += len(chunk)
        if seen >= pp["avazu_rows"]:
            break
    df = pd.concat(chunks, ignore_index=True).iloc[: pp["avazu_rows"]]
    df = df.rename(columns={label_col: "click"})
    manifest["avazu_rows_read"] = len(df)
    manifest["avazu_empirical_ctr_overall"] = float(df["click"].mean())
    AVAZU_CATEGORICAL[:] = categorical

    # An "advertisement" is a unique combination of the categorical fields.
    df["ad_key"] = df[AVAZU_CATEGORICAL].agg("|".join, axis=1)
    grp = df.groupby("ad_key")["click"].agg(["mean", "count"])
    manifest["avazu_unique_ads"] = len(grp)

    # Keep only ads with enough impressions for a stable empirical CTR.
    grp = grp[grp["count"] >= 30].copy()
    manifest["avazu_ads_after_min_impressions"] = len(grp)

    ads = grp.reset_index().rename(
        columns={"mean": "empirical_ctr", "count": "impressions"})

    k, f = pp["n_topics"], rep["n_hash_features"]
    ads["topic"] = [stable_hash(key, k) for key in ads["ad_key"]]
    # Trust prior: empirical CTR rescaled to [0, 1] across the inventory.
    lo, hi = ads["empirical_ctr"].min(), ads["empirical_ctr"].max()
    ads["trust_prior"] = ((ads["empirical_ctr"] - lo) / (hi - lo + 1e-9))
    # Intrusiveness: banner position, min-max scaled. The index is looked up
    # by name rather than hard-coded, so it stays correct if the field list
    # in AVAZU_CATEGORICAL is ever reordered.
    banner_idx = AVAZU_CATEGORICAL.index(intrusive_col)
    banner = pd.to_numeric(ads["ad_key"].str.split("|").str[banner_idx],
                           errors="coerce")
    if banner.isna().all():
        raise ValueError(
            f"Could not read numeric values from the {intrusive_col} field; "
            "the intrusiveness variable cannot be constructed.")
    banner = banner.fillna(banner.median())
    ads["intrusiveness"] = ((banner - banner.min())
                            / (banner.max() - banner.min() + 1e-9))
    manifest["avazu_intrusiveness_field"] = intrusive_col
    # Hashed representation of the categorical fields.
    for j in range(f):
        ads[f"h{j}"] = [
            1.0 if stable_hash(f"{key}#{j}", 2) else 0.0 for key in ads["ad_key"]
        ]

    manifest["ad_pool_final"] = len(ads)
    return ads


def build_synthetic(cfg: dict, out: Path, manifest: dict) -> None:
    """Stand-in data so the pipeline can be exercised without the downloads.

    FOR PIPELINE TESTING ONLY. Numbers produced from synthetic data are
    meaningless and must never be reported. Every file written here is
    stamped ``synthetic: true`` in the manifest.
    """
    pp, rep = cfg["preprocess"], cfg["representation"]
    rng = np.random.default_rng(cfg["seed"])
    k, n = pp["n_topics"], 5000

    words = ["great", "awful", "today", "phone", "delivery", "price", "again",
             "never", "love", "hate", "quick", "slow", "cheap", "broken"]
    texts = [" ".join(rng.choice(words, size=rng.integers(5, 20)))
             + f" #{i}" for i in range(n)]
    pool = pd.DataFrame({
        "text_clean": texts,
        "sentiment": rng.choice([-1.0, 1.0], size=n),
        "arousal": rng.random(n).astype(float),
        "topic": [stable_hash(t, k) for t in texts],
        "language": rng.choice(["en", "zh"], size=n),
    })
    pool.to_csv(out / "sentiment_pool.csv.gz", index=False, compression="gzip")

    m, f = 800, rep["n_hash_features"]
    keys = [f"synthetic-ad-{i}" for i in range(m)]
    ads = pd.DataFrame({
        "ad_key": keys,
        "empirical_ctr": rng.beta(2, 40, size=m),
        "impressions": rng.integers(30, 5000, size=m),
        "topic": [stable_hash(kk, k) for kk in keys],
        "trust_prior": rng.random(m),
        "intrusiveness": rng.random(m),
    })
    for j in range(f):
        ads[f"h{j}"] = [1.0 if stable_hash(f"{kk}#{j}", 2) else 0.0 for kk in keys]
    ads.to_csv(out / "ad_pool.csv.gz", index=False, compression="gzip")

    manifest.update({"synthetic": True, "sentiment_pool_final": n,
                     "ad_pool_final": m})
    print(f"  -> synthetic pools written ({n} utterances, {m} ads)")
    print("  !! synthetic data: results from this are NOT reportable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="generate stand-in data to test the pipeline; "
                         "results from it must never be reported")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    raw = ROOT / cfg["paths"]["raw"]
    out = resolve(cfg, "data")

    manifest: dict = {"config_sha256": cfg["_config_sha256"], "seed": cfg["seed"],
                      "synthetic": False}

    if args.synthetic:
        print("Building SYNTHETIC pools (pipeline test only) ...")
        build_synthetic(cfg, out, manifest)
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        return

    print("Building sentiment pool ...")
    pool = build_sentiment_pool(cfg, raw, manifest)
    pool.to_csv(out / "sentiment_pool.csv.gz", index=False, compression="gzip")
    print(f"  -> {len(pool):,} utterances")

    print("Building ad pool ...")
    ads = build_ad_pool(cfg, raw, manifest)
    ads.to_csv(out / "ad_pool.csv.gz", index=False, compression="gzip")
    print(f"  -> {len(ads):,} advertisements")

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nRecord counts written to {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
