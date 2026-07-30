# Supplementary Information File S1 — data, code and materials

Reproduction package for "A credibility-aware and context-sensitive
framework for social media advertising delivery".

> ## Read this first
>
> This package is a **fresh, complete implementation of the method exactly
> as described in the manuscript**. It was written to restore the analysis
> pipeline, and it has **not** been calibrated to reproduce the specific
> numbers currently printed in Tables 1 and 2 or Figs. 2 and 3.
>
> Run it, then **replace the numbers in the manuscript with the numbers
> this code actually produces**, and state the reported seed. Do not submit
> this package alongside the existing numbers unless they match — a
> reviewer who runs the code will compare them, and reporting figures that
> the supplied code does not generate is a research-integrity problem, not
> a formatting one.
>
> Every simulator parameter is in `config.yaml` and every modelling
> assumption is documented in `VARIABLES.md`. Nothing is hidden in the
> code, and no result is hard-coded.

## 1. What is in this package

```
supplementary/
├── README.md          this file — how to run, software versions
├── VARIABLES.md       variable definitions, coding scheme, transformations,
│                      metric formulas, and every simulator assumption
├── requirements.txt   pinned dependencies
├── config.yaml        all hyperparameters, weights and random seeds
├── prompts/
│   ├── user_agent_system.txt   system prompt given to every LLM user agent
│   └── user_agent_turn.txt     per-impression prompt template
├── src/
│   ├── preprocess.py    build cleaned datasets from the three public sources
│   ├── env.py           the advertising delivery simulator
│   ├── model.py         Deep Sentiment Network (interest + credibility heads)
│   ├── policies.py      the proposed policy and all baselines, incl. the RL agent
│   ├── agents.py        LLM user agents + offline heuristic agent
│   ├── metrics.py       CTR, NDCG, CRR and UFI
│   ├── run_benchmark.py   -> Table 1
│   ├── run_llm_eval.py    -> Table 2 and Fig. 2
│   └── run_robustness.py  -> Fig. 3
├── data/              (created by preprocess.py — cleaned datasets)
└── results/           (created by the run_* scripts — logs, tables, figures)
```

## 2. Software environment

Tested with Python 3.10 and 3.11.

```
pip install -r requirements.txt
```

| package | version tested |
|---------|----------------|
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| torch | 2.3.1 (CPU is sufficient) |
| matplotlib | 3.8.4 |
| pyyaml | 6.0.1 |
| requests | 2.32.3 (only needed for the LLM conditions) |

## 3. Getting the source data

Download these three public datasets and put them in `raw/`:

| dataset | file expected in `raw/` | source |
|---------|------------------------|--------|
| Sentiment140 | `training.1600000.processed.noemoticon.csv` | http://help.sentiment140.com/for-students |
| Weibo_Senti_100k | `weibo_senti_100k.csv` | https://github.com/SophonPlus/ChineseNlpCorpus |
| Avazu CTR Prediction | `train.csv` (or `train.gz`) | https://www.kaggle.com/c/avazu-ctr-prediction |

Avazu's `train.csv` is ~6 GB; `preprocess.py` streams it and keeps only the
first `avazu_rows` records (default 2,000,000), so you do not need to load
it all into memory.

## 4. Running everything

```bash
cd supplementary
python src/preprocess.py                 # writes data/*.csv.gz + data/manifest.json
python src/run_benchmark.py              # -> results/table1_benchmark.csv
python src/run_llm_eval.py --agent heuristic   # -> results/table2_llm.csv, results/fig2.pdf
python src/run_robustness.py             # -> results/fig3.pdf, results/robustness.csv
```

Each script writes a JSON sidecar recording the config hash, the seed and
the package versions used, so that any run can be traced.

### Running the actual LLM user agents

`--agent heuristic` uses the offline rule-based user model and needs no API
keys; it is fully deterministic and is what a reviewer can run without
cost. To reproduce the LLM user-agent conditions of Table 2:

```bash
export OPENAI_API_KEY=...      # GPT-4o, GPT-5o
export DEEPSEEK_API_KEY=...    # DeepSeek-V3
export MOONSHOT_API_KEY=...    # Kimi-2
export GEMINI_API_KEY=...      # Gemini-1.5 Pro
python src/run_llm_eval.py --agent llm --models gpt-4o,gpt-5o,gemini-1.5-pro,deepseek-v3,kimi-2
```

Responses are cached in `results/llm_cache/` keyed by a hash of the prompt,
so a re-run costs nothing and the exact model outputs behind every reported
number are preserved for inspection. **Ship the cache with the submission**
— it is the audit trail for the LLM conditions.

## 5. Before you upload this package

- [ ] Run all four scripts end to end on a clean checkout.
- [ ] Copy the produced numbers into Tables 1 and 2 and regenerate Figs. 2 and 3.
- [ ] Include `data/`, `results/` and `results/llm_cache/` in the upload.
- [ ] Confirm no author name, institution or personal repository URL appears
      anywhere in this folder (the journal's peer review is double-anonymised
      and reviewers can see supplementary files).
- [ ] Zip the folder as `Supplementary_Information_S1.zip`.
