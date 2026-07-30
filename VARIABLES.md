# Variable definitions, coding scheme, transformations and formulas

Companion to Supplementary Information File S1. Everything the editorial
office asked for under "variable definitions, coding schemes and applied
transformations" and "formulas, statistical code and scripts" is here.

Parameter values are not repeated in this file — they all live in
`config.yaml`, so that there is exactly one place to look.

---

## 1. Source datasets and the coding scheme applied to them

| source | field used | original coding | coding applied here |
|--------|-----------|-----------------|---------------------|
| Sentiment140 | `target` | 0 = negative, 2 = neutral, 4 = positive | rows with `target` ∈ {0, 4} kept; `sentiment` = +1 if 4, −1 if 0. Neutral rows are dropped, because the environment's mood-congruence term is defined on signed valence. |
| Sentiment140 | `text` | raw tweet | cleaned (§2), becomes `text_clean` |
| Weibo_Senti_100k | `label` | 1 = positive, 0 = negative | `sentiment` = +1 if 1, −1 if 0 |
| Weibo_Senti_100k | `review` | raw post | cleaned (§2), becomes `text_clean` |
| Avazu | `click` | 0/1 per impression | aggregated to an empirical CTR per advertisement |
| Avazu | `C1`, `banner_pos`, `site_category`, `app_category`, `device_type`, `device_conn_type`, `C15`, `C16`, `C18` | anonymised categorical codes | concatenated into `ad_key`, which defines one "advertisement" |

## 2. Text transformations, in the order applied

1. URLs (`https?://…`, `www.…`) replaced with a space.
2. `@mentions` replaced with a space.
3. Character runs of length ≥ 3 collapsed to 2 (`sooooo` → `soo`).
4. Whitespace collapsed and trimmed.
5. Truncated to `preprocess.max_text_chars`.
6. Rows shorter than `preprocess.min_text_chars` dropped.
7. Exact duplicates on `text_clean` dropped.

Counts before and after steps 6 and 7 are written to `data/manifest.json`.

The same pipeline is applied to English and Chinese text. No tokeniser or
stop-word list is language-specific, because the representation used
downstream is a hash of the cleaned string rather than a bag of words.

## 3. Derived variables

### 3.1 User-side

| variable | range | definition |
|----------|-------|------------|
| `sentiment` | {−1, +1} | valence, from the dataset label (§1) |
| `arousal` | [0, 1] | **proxy**: `0.5·marks + 0.3·caps + 0.2·length`, where `marks` = min(count of `!?！？`, 5)/5, `caps` = share of upper-case among alphabetic characters, `length` = min(chars, 280)/280. This is a surface-form intensity proxy, **not** a validated psychometric arousal measure. It is reported as such in the Methods section. |
| `topic` | {0 … K−1} | deterministic MD5 hash bucket of `text_clean`. Stands in for an unsupervised topic assignment; it is stable across runs and machines (Python's built-in `hash()` is salted per process and is not used). |
| `interest` | simplex over K | Dirichlet draw with α = 0.4 everywhere and α = 3.0 on the utterance's topic, so interest is concentrated but not degenerate |
| `trust` | [0, 1] | starts at `environment.trust_init`, evolves by §4.2 |
| `fatigue` | [0, 1] | starts at 0, evolves by §4.3 |

The user state vector is `[sentiment, arousal, interest(K), trust, fatigue]`,
of length K + 4.

### 3.2 Advertisement-side

| variable | range | definition |
|----------|-------|------------|
| `empirical_ctr` | [0, 1] | mean `click` over all Avazu impressions sharing that `ad_key`; advertisements with fewer than 30 impressions are dropped |
| `trust_prior` | [0, 1] | `empirical_ctr` min–max rescaled across the inventory. **Assumption:** historically better-performing inventory is treated as more established/trusted. This is a proxy, not a measure of trust. |
| `intrusiveness` | [0, 1] | `banner_pos` min–max rescaled |
| `topic` | {0 … K−1} | hash bucket of `ad_key` |
| `h0 … h{F−1}` | {0, 1} | hashed representation of the categorical fields |

The ad vector is `[topic one-hot(K), trust_prior, intrusiveness, h0…h{F−1}]`.

## 4. The simulator

Everything in this section is a **modelling assumption that defines the
environment**. None of it is an empirical finding, and none of it is
estimated from data. The coefficients are in `config.yaml`.

### 4.1 Click model

```
relevance(u, a)        = u.interest[a.topic]
mood_congruence(u, a)  = a.trust_prior            if u.sentiment >= 0
                       = 1 - a.intrusiveness      if u.sentiment <  0

p_click = sigmoid( click_bias
                 + w_relevance * relevance
                 + w_trust     * u.trust
                 + w_mood      * mood_congruence
                 - w_fatigue   * u.fatigue )
```

The mood-congruence branch encodes the assumption that positively disposed
users are more receptive to established inventory, while negatively
disposed users react mainly to how disruptive the placement is. This is the
single most consequential assumption in the simulator and should be stated
as a limitation.

### 4.2 Trust dynamics

```
on click               : trust += trust_gain_on_click
on no-click AND
  relevance < mismatch_threshold
                       : trust -= trust_loss_on_mismatch,  negative feedback
more than overexposure_limit impressions
  within overexposure_window steps
                       : trust -= trust_loss_on_overexposure, negative feedback
trust clipped to [trust_floor, trust_ceiling]
```

### 4.3 Fatigue dynamics

```
on each impression : fatigue += fatigue_per_impression   (capped at 1)
on each step       : fatigue -= fatigue_decay_per_step   (floored at 0)
```

### 4.4 Interest drift

At the end of every step, with probability `interest_drift_prob`, the user's
interest vector is redrawn around a uniformly random new topic. This is the
parameter swept in the robustness experiment.

## 5. Model

Deep Sentiment Network: shared trunk
`Linear(d_u + d_a → d_hidden) → ReLU → Dropout → Linear(d_hidden → d_hidden) → ReLU`,
then two independent `Linear(d_hidden → 1) → sigmoid` heads.

Training targets, per impression:

- interest head → the observed click (0/1)
- credibility head → 1 if trust did not fall on that impression, else 0

Loss = BCE(interest) + BCE(credibility), Adam, parameters in `config.yaml`.

Training data come from a **uniformly random exploration policy**, not from
the policy being evaluated. Training on the proposed policy's own
impressions would bias the interest head in its favour.

## 6. Decision rules

```
joint matching score      M = alpha * S_interest + (1 - alpha) * S_credibility   (Eq. 5)
RL state                  s = [user vector, M of the selected ad]                (Eq. 6)
RL reward                 r = lambda * engagement - mu * fatigue                 (Eq. 7)
    engagement = 1 if the impression was clicked, else 0
    fatigue    = increase in the user's fatigue level on that step
                 + 1 if a negative-feedback event occurred
```

The show/skip policy is a DQN with a target network, experience replay and
linear ε decay; all parameters are under `rl:` in `config.yaml`. Every
learning policy is trained with exploration on, then **evaluated with
exploration off**, so reported numbers come from a fixed policy.

## 7. Metric formulas

Let *I* be the number of impressions actually shown and *S* the number of
steps (impressions + skips).

**CTR** = clicks / *I*. Skipped steps are excluded from the denominator; a
policy that skips is not penalised twice (once here, once in UFI).

**NDCG@k** = DCG@k / IDCG@k, with
DCG@k = Σᵢ₌₁ᵏ gainᵢ / log₂(i + 1),
gain = the environment's graded relevance
`relevance(u, a) × (0.5 + 0.5 × a.trust_prior)`,
and IDCG@k the same quantity under the ideal ordering of that candidate set.
k = `metrics.ndcg_k`. Computed per step over the candidate set and averaged.

**CRR** = mean over sessions of `trust_T / trust_0`. Not clipped: a value
above 1 means the session ended more trusting than it began.

**UFI** = `w_skip · user_skip_rate + w_neg · negative_rate + w_decay · decay`,
where
user_skip_rate = (*I* − clicks) / *I* — the share of **shown** advertisements
the user scrolled past. This is a user behaviour. It is deliberately *not*
the rate at which the policy withheld an advertisement: counting that here
would penalise a policy for protecting the user from over-exposure, which is
the behaviour the index exists to reward. The policy's own withhold rate is
still logged, as `_policy_withhold_rate`, so it can be inspected.
negative_rate = negative-feedback events / *I*,
decay = mean over sessions of `max(0, (r_first − r_second) / r_first)` with
`r_first`/`r_second` the interaction rate in the first and second half of the
session (sessions with fewer than 4 impressions are excluded).
The three weights sum to 1, so UFI ∈ [0, 1] and **lower is better**.

## 8. Statistical reporting

`run_benchmark.py` and `run_robustness.py` take a `--repeats` argument and
report **mean ± standard deviation across independent seeds**. Use those,
not a single run — the journal's statistical guidelines require a measure of
centre and a measure of variability for every reported quantity, and require
that any use of the word "significant" be backed by an actual *P* value.

If you want a significance test between two policies, the appropriate
comparison is over the per-repeat values in `table1_raw_runs.csv` (paired by
seed), not over individual sessions, which are not independent within a run.

## 9. Large language model user agents

- Prompts: `prompts/user_agent_system.txt` and `prompts/user_agent_turn.txt`,
  used verbatim and identically for every model.
- Decoding: temperature and token limit from `config.yaml`; identical across
  models.
- No web search, no tool use — enforced by the API calls themselves, which
  expose no tools.
- The agent decides the click; the environment still applies the trust and
  fatigue dynamics, so that the LLM and offline conditions are accounted for
  identically and remain comparable on CRR and UFI.
- Every response is cached at `results/llm_cache/<model>_<sha256>.json`,
  storing the exact prompt, the raw completion and the parsed judgement.
- Responses that fail to parse are recorded as a non-click with neutral
  credibility and flagged `parse_failed`, so parse failures appear in the
  results instead of silently biasing them. **Report the parse-failure rate.**
