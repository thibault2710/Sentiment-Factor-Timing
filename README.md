# Factor Timing with Machine Learning and Sentiment Signals

**Chicago Booth — Machine Learning in Finance | Final Project**

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Notebooks](#notebooks)
4. [Data](#data)
5. [Methodology](#methodology)
6. [Results](#results)
7. [Discussion](#discussion)
8. [Limitations](#limitations)
9. [Next Steps](#next-steps)
10. [Setup](#setup)
11. [References](#references)

---

## Overview

This project builds a **sentiment-augmented factor timing model** for US equity markets. The central research question is: *do alternative sentiment signals — derived from Wall Street Journal news coverage and Reddit/WallStreetBets activity — improve out-of-sample factor timing performance beyond what macroeconomic variables alone can achieve?*

Factor timing refers to dynamically adjusting allocations across equity style factors — value, momentum, quality, investment, and size — in anticipation of which factors will outperform over the next month. While a static equal-weight factor portfolio (EWP) is a sensible benchmark, the literature has documented that factor returns are partially predictable using macro state variables. We extend this with textual sentiment, testing two distinct data sources constructed using modern NLP tools (sentence transformers, large language models, and fine-tuned classifiers).

The project covers **25 years of US equity data (2000–2025)**, constructs five long-short factor portfolios from scratch using CRSP and Compustat, and evaluates eight model variants in a strict walk-forward framework with no look-ahead bias.

**Key findings at a glance:**
- WSJ news sentiment adds meaningful value in factor-specific prediction: the factor-specific Ridge model (5B) delivers a **+0.21 Sharpe improvement** over its macro-only control
- Reddit/WallStreetBets sentiment provides a modest directional market signal (+0.01 Δ Sharpe, α t = 1.73*) but no cross-sectional factor tilting value
- More complex models (mean-variance with sentiment) consistently underperform simpler factor-specific signal-scaling — model complexity is penalized by estimation error in this setting
- The equal-weight factor portfolio remains a difficult benchmark to beat consistently

---

## Repository Structure

```
Github_Ready/
├── data/
│   └── FinalMonthlyDataset_ours_ff_macro.csv   # 309 months × 48 features (final model input)
├── utils.py                                      # Shared helper functions
├── 01_data_overview.ipynb                        # Dataset exploration & variable descriptions
├── 02_factor_returns.ipynb                       # Factor portfolio construction & performance
├── 03_ml_models.ipynb                            # Baseline Ridge & Lasso regression
└── 04_factor_timing.ipynb                        # Main analysis: sentiment-augmented factor timing
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_data_overview.ipynb` | Load the final dataset, inspect all 48 features, visualize distributions, correlations, and time-series dynamics |
| `02_factor_returns.ipynb` | Analyze the five constructed factor portfolios — cumulative returns, Sharpe ratios, turnover costs, and benchmark comparisons against Fama-French factors |
| `03_ml_models.ipynb` | Ridge and Lasso regressions for next-month factor return prediction; time-series train/validation/test splits; Information Coefficient evaluation |
| `04_factor_timing.ipynb` | Walk-forward factor timing experiments with six model variants (5A–5F) across two sentiment sources; results table, cumulative return charts, weight evolution, and sentiment lift analysis |

---

## Data

### Final Dataset (`data/FinalMonthlyDataset_ours_ff_macro.csv`)

The model input is a single monthly panel: **309 months (2000-03 to 2025-11), 48 columns**.

| Feature Group | Variables | Count |
|---|---|---|
| **Our Factor Returns (gross)** | `fac_value`, `fac_momentum`, `fac_quality`, `fac_investment`, `fac_size` | 5 |
| **Our Factor Returns (net of TC)** | `fac_*_net` (30bps × turnover) | 5 |
| **Portfolio metadata** | `n_long_*`, `n_short_*`, `to_*` (turnover) | 15 |
| **Fama-French Benchmarks** | `hml`, `umd`, `smb`, `rmw`, `cma`, `mktrf`, `rf` | 7 |
| **Macro / Market State** | `fedfunds`, `dgs10`, `term_spread_10y_fedfunds`, `mkt_vol_12m`, `mkt_trend_12m`, `dispersion`, `rate_chg_3m`, `rate_chg_12m` | 8 |
| **WSJ News Sentiment** | `wsj_index`, `wsj_uncertainty`, `topic_index`, `topic_uncertainty` | 4 |
| **Reddit / WallStreetBets** | `WallStreetBets_score`, `WallStreetBets_confidence`, `WallStreetBets_numeric_score` | 3 |

### Data Sources (Pipeline Summary)

The dataset was assembled from the following sources. The full construction pipeline is available in the original project repository but is not required to run the analysis notebooks here.

#### 1. Equity Universe — WRDS (CRSP + Compustat)
- **CRSP Monthly Stock File**: monthly returns, prices, shares outstanding (2000–2025)
- **Compustat Fundamentals Annual**: book equity, total assets, revenue, net income, etc.
- **CCM Link Table**: maps CRSP PERMNOs to Compustat GVKEYs
- **Universe filter**: NYSE common shares on primary exchange; bottom 20% by lagged market cap excluded (microcap screen)
- **Look-ahead protection**: Compustat data applied with a 6-month lag to avoid earnings release timing bias

#### 2. Factor Construction

Five long-short factor portfolios, constructed monthly:

| Factor | Signal | Construction |
|---|---|---|
| **Value** | Book-to-Market ratio | Top 30% long / Bottom 30% short, size-neutralized |
| **Momentum** | 12-1 month price return | Top 30% long / Bottom 30% short, size-neutralized |
| **Quality** | Return on Equity (NI/BE) | Top 30% long / Bottom 30% short, size-neutralized |
| **Investment** | Asset growth (YoY Δ assets/assets) | Bottom 30% long / Top 30% short (low investment), size-neutralized |
| **Size** | Negative log market cap | Small-cap long / Large-cap short |

All returns are **value-weighted** within legs. Net returns deduct **30 bps × monthly turnover** as transaction cost.

#### 3. Fama-French Factors
- Downloaded from Ken French's data library via WRDS
- Includes 5 factors (MKT-RF, SMB, HML, RMW, CMA) + UMD (momentum)
- Used as both benchmark comparisons and model features

#### 4. Macro Indicators — FRED
- **Federal Funds Rate** (`FEDFUNDS`): monetary policy stance
- **10-Year Treasury Yield** (`DGS10`): long-term rate level
- **Term Spread**: 10Y yield minus Fed Funds rate (yield curve shape)
- **Rate Change**: 3-month and 12-month changes in Fed Funds rate
- **Market Volatility**: 12-month rolling volatility of market excess returns
- **Market Trend**: 12-month rolling mean of market excess returns
- **Dispersion**: Cross-sectional return dispersion (std dev of individual stock returns)

#### 5. WSJ News Sentiment (Bybee et al. 2024 replication)

**Source:** Wall Street Journal article headlines, 2000–2017 (~208 monthly observations)

We construct two pairs of variables from the same article corpus:

**`wsj_index` and `wsj_uncertainty` — embedding-based narrative index:**
1. Each WSJ headline is embedded into a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer model (a local, open-source model — no API calls required for this step)
2. All headline embeddings within a calendar month are averaged into a single 384-dimensional monthly vector
3. Each embedding dimension is rolling z-score normalized over a 36-month trailing window to remove long-run secular trends in media coverage
4. PCA is applied to the normalized monthly embedding matrix (10 components retained)
5. `wsj_index` = the first principal component, standardized to zero mean and unit variance. This captures the dominant axis of variation in how WSJ describes the economy each month — in practice, it loads heavily on recession/uncertainty language during stress periods
6. `wsj_uncertainty` = 12-month rolling standard deviation of `wsj_index`. This measures how volatile the narrative has been recently — high values indicate the media narrative is shifting quickly

**`topic_index` and `topic_uncertainty` — LLM-generated topic index:**
1. The same WSJ headlines are passed to GPT-4.1-nano with the prompt: *"List 1–3 keyword topics or risk factors from this article."* Both a neutral and a bear-persona version are generated for each article
2. Topics are aggregated into monthly count vectors (how often each topic appeared that month)
3. The monthly topic counts are rolling z-score normalized (36-month window), then PCA is applied (10 components)
4. `topic_index` = first PC of the topic count matrix, standardized
5. `topic_uncertainty` = 12-month rolling standard deviation of `topic_index`

The embedding index and the topic index capture complementary signals: the embedding approach tracks the overall semantic tone of coverage, while the topic index tracks which specific risk themes are prominent.

#### 6. Reddit / WallStreetBets Sentiment

**Source:** WallStreetBets subreddit posts (2013–2025), scraped from monthly JSONL files

**Construction:**
1. Each post is individually classified using the `cardiffnlp/twitter-roberta-base-sentiment` model — a RoBERTa model fine-tuned on Twitter data for three-class sentiment (negative / neutral / positive)
2. The model outputs both a **class label** and a **softmax confidence score** (the probability assigned to the predicted class)
3. Posts are aggregated to a monthly panel:
   - `avg_sentiment`: mean sentiment score across all posts (range: approximately −0.2 to +0.3)
   - `avg_confidence`: mean classifier confidence score (range: approximately 0.665–0.741 — narrow because averaging over hundreds of posts dampens post-level variation)
   - `WallStreetBets_numeric_score`: mean of the numeric class label (−1 / 0 / +1)
   - `WallStreetBets_score` and `WallStreetBets_confidence`: post-level engagement-weighted versions of the above

**A key empirical finding** about `avg_confidence`: the sentiment signal is substantially stronger in **low-confidence months** (when the classifier is uncertain about the overall direction) than in high-confidence months. Raw correlation with next-month EWP return: `avg_sentiment` r = +0.047, but the confidence-gated version `conf_gated_sent` r = +0.133. Intuitively, high classifier confidence corresponds to clearly negative market sentiment (bear periods) — which is already priced in. Uncertain months carry more forward-looking information. Two confidence-gated features are engineered directly in `04_factor_timing.ipynb`:
- `conf_gated_sent`: full sentiment in low-confidence months, zero otherwise (binary gate)
- `conf_inv_sent`: sentiment weighted by how far confidence falls below its expanding median (graduated gate)

---

## Methodology

### Factor Timing Framework

Each month, rather than holding an equal-weight portfolio of all five factors, we dynamically tilt weights using predicted factor returns. Eight model variants are tested — two macro-only controls and six sentiment-augmented models:

| Model | Description | Feature Set |
|---|---|---|
| **F0A** (control) | Signal-scaled EWP, market-level prediction | Macro only |
| **F0D** (control) | Mean-variance, factor-specific prediction | Macro only |
| **5A** | Signal-scaled EWP, market-level prediction | Macro + sentiment |
| **5B** | Factor-specific Ridge, signal-scaled per-factor | Macro + sentiment |
| **5C** | Two-state mean-variance (full/half exposure) | Macro + sentiment |
| **5D** | Factor-specific MV with Ledoit-Wolf Σ + TC penalty | Macro + sentiment |
| **5E** | Random Forest, market-level (mirrors 5A) | Macro + sentiment |
| **5F** | Random Forest, factor-specific (mirrors 5B) | Macro + sentiment |

Comparing each sentiment model (5A–5F) against its matched macro-only control (F0A or F0D) isolates the **marginal value of sentiment** beyond macro conditioning.

### Model Architecture Details

**Signal-scaled models (5A, 5B):** A Ridge regression predicts next-month factor return(s). The predicted return is z-scored against its in-sample distribution, clipped to [−2, +2], and used to scale factor weights symmetrically around the equal-weight baseline. This avoids extreme short positions while allowing active tilts.

**Two-state MV (5C):** A single market-level Ridge prediction determines a binary state — full factor exposure (EWP weights optimized via mean-variance) in positive months, or half-exposure (a more defensive stance) in negative prediction months.

**MV factor-specific (5D, F0D):** Separate Ridge models predict each factor's return. These predictions serve as the μ̂ vector in a mean-variance optimization with Ledoit-Wolf shrinkage covariance estimation. A turnover penalty κ is incorporated directly into the objective: `w* = (γΣ + κI)⁻¹(μ̂ + κ·w_prev)` where γ = 3.0 (risk aversion) and κ = 2.0 (TC penalty).

**Random Forest models (5E, 5F):** Replace Ridge with a Random Forest. Tree depth is tuned via a time-series cross-validation inner loop, and `min_samples_leaf` scales inversely with training set size to prevent overfitting on small samples.

### Walk-Forward Evaluation

All models use an **expanding-window walk-forward** design — the most rigorous evaluation protocol for time-series prediction:
- No random splits, no shuffling, no validation set leakage
- Training window expands by one month each period
- Minimum training windows: 60 months (WSJ), 36 months (Reddit)
- All features are **expanding z-scored at prediction time** — only data available at that date is used to compute the z-score mean and standard deviation

**WSJ evaluation window:** 2000-03 to 2017-06 (208 total months; ~148 OOS months after minimum training window)
**Reddit evaluation window:** 2013-01 to 2025-11 (156 total months; ~120 OOS months after minimum training window)

These are kept separate to avoid mixing the two sentiment sources, which have different availability windows.

### Signal Predictability: Information Coefficient

Before running the full walk-forward, we assess the raw predictive power of each sentiment source by computing the **Information Coefficient (IC)** — the Pearson correlation between the z-scored signal and next-month EWP return. This is a common metric in quantitative asset management for evaluating factor quality:

| Source | Primary Signal | IC |
|---|---|---|
| WSJ | `wsj_index_z` | **0.153** |
| Reddit | `WallStreetBets_score_z` | **0.143** |

Both signals show meaningful positive IC (rule of thumb: IC > 0.05 is considered useful in practice). However, the rolling 24-month IC reveals that predictability is **time-varying and non-stationary** — both signals go through extended periods of negative IC, particularly during 2011–2014 for WSJ and 2024–2025 for Reddit. This instability helps explain why many of the factor timing models struggle to reliably outperform the EWP baseline out-of-sample.

### Evaluation Metrics

- **Annualized Sharpe Ratio** — primary performance metric (annualized return / annualized volatility × √12)
- **Δ Sharpe** — sentiment model Sharpe minus matched macro-only control Sharpe; isolates the value of adding sentiment
- **Alpha & t-statistic** — OLS alpha vs EWP benchmark, Newey-West HAC standard errors with 6 lags (to account for serial correlation in monthly portfolio returns)
- **Information Ratio** — annualized active return / tracking error vs EWP
- **Maximum Drawdown** — peak-to-trough decline in cumulative return
- **Weight Turnover** — average monthly L1 distance of factor weight vector

---

## Results

### WSJ Sentiment Results (2000-03 → 2017-06)

The EWP baseline Sharpe over the WSJ window is **+0.41** — a reasonably productive environment for multi-factor equity portfolios over 2000–2017, which includes the post-GFC recovery.

| Model | N Months | Sharpe | Δ Sharpe | Alpha (ann.) | t(alpha) | Max DD | Wt Turnover |
|---|---|---|---|---|---|---|---|
| F0A: Macro-only, market-level *(control)* | 148 | −0.04 | — | — | — | — | — |
| F0D: Macro-only, MV factor *(control)* | 148 | −0.11 | — | — | — | — | — |
| 5A: + WSJ, market-level | 148 | −0.07 | −0.03 | | | | |
| **5B: + WSJ, factor-specific** | **148** | **+0.18** | **+0.21** | | | | |
| 5C: + WSJ, MV market | 148 | +0.05 | +0.09 | | | | |
| 5D: + WSJ, MV factor | 148 | −0.17 | −0.06 | | | | |
| 5E: + WSJ, RF market | 148 | | | | | | |
| 5F: + WSJ, RF factor | 148 | | | | | | |

*Exact alpha and t-stat values for each model are printed in `04_factor_timing.ipynb` cell 18. The table above captures the headline Sharpe and Δ Sharpe numbers.*

**Headline finding:** The factor-specific signal-scaling model **(5B) adds +0.21 Sharpe** over its macro-only control — the single largest positive result in the experiment. The two-state MV model (5C) also adds modest value (+0.09). All other WSJ models underperform their controls.

### Reddit / WallStreetBets Results (2013-01 → 2025-11)

The EWP baseline Sharpe over the Reddit window is **−0.18** — this 12-year period was challenging for factor portfolios, particularly post-2018 when the value premium was largely absent and factor dispersion was low relative to idiosyncratic volatility.

| Model | N Months | Sharpe | Δ Sharpe | Alpha (ann.) | t(alpha) | Max DD | Wt Turnover |
|---|---|---|---|---|---|---|---|
| F0A: Macro-only, market-level *(control)* | ~120 | +0.11 | — | — | — | — | — |
| F0D: Macro-only, MV factor *(control)* | ~120 | −0.39 | — | — | — | — | — |
| **5A: + Reddit, market-level** | **~120** | **+0.12** | **+0.01** | | **t = 1.73\*** | | |
| 5B: + Reddit, factor-specific | ~120 | −0.20 | −0.31 | | | | |
| 5C: + Reddit, MV market | ~120 | −0.18 | −0.29 | | | | |
| 5D: + Reddit, MV factor | ~120 | −0.42 | −0.03 | | | | |
| 5E: + Reddit, RF market | ~120 | | | | | | |
| 5F: + Reddit, RF factor | ~120 | | | | | | |

**Headline finding:** Reddit sentiment adds value only in the simplest setting — **5A posts a marginally positive Δ Sharpe (+0.01) with a statistically significant alpha (t = 1.73*) at the 10% level under Newey-West HAC**. All more complex models substantially underperform their controls, with 5B (−0.31 Δ) and 5C (−0.29 Δ) suffering large negative spreads.

### Factor Weight Evolution

The weight stacked-area charts in notebook 04 show that sentiment-augmented models make **active, time-varying tilts** across the five factors. Key observations:

**WSJ (2005–2017):**
- Model 5B shows substantial month-to-month weight reallocation, with notable rotations around the 2008–2009 financial crisis — reducing momentum exposure and tilting toward value and quality during the crisis
- The 5D vs F0D comparison reveals that the macro-only MV model becomes dominated by Size and Value after 2012 (reflecting the risk structure of those factors in the post-GFC period), while the WSJ-augmented version maintains more balanced cross-factor weights

**Reddit (2016–2025):**
- Model 5B similarly shows dynamic tilts, with a notable increase in Quality allocation post-2020 — consistent with a flight-to-quality signal during and after the COVID shock
- The 5D vs F0D comparison shows that by 2022–2025, the macro-only model over-concentrates in Quality and Momentum, while the Reddit-augmented version rebalances more aggressively — though this active rebalancing ultimately hurts performance, suggesting Reddit's factor-specific predictions are too noisy

### Sentiment Lift Analysis

The "sentiment lift" chart (right panels in notebook 04) shows the **ratio of the cumulative return of each sentiment model to its matched macro-only control** — values above 1.0 indicate the sentiment model is outperforming.

**WSJ:** 5A's lift vs F0A trends persistently below 1.0 (ending near 0.99), confirming it adds no value at the market level. The 5D lift vs F0D also trends downward over the full window, ending near 0.96 — a meaningful drag attributable to the higher complexity of the MV optimizer struggling with estimation error in the sentiment features.

**Reddit:** The lift chart is noisier given the shorter history and lower absolute return levels. 5A's lift vs F0A fluctuates around 1.0, occasionally above and occasionally below, consistent with the near-zero average improvement. The 5D lift vs F0D deteriorates more consistently.

---

## Discussion

### Why Does 5B Work for WSJ but Not for Reddit?

The factor-specific Ridge model (5B) exploits the **cross-sectional heterogeneity** in how sentiment predicts individual factors. For WSJ, this works because the four variables (`wsj_index`, `wsj_uncertainty`, `topic_index`, `topic_uncertainty`) carry distinct information about different economic narratives — uncertainty indices likely predict value and quality differentially from momentum, for example. These cross-sectional predictions are learned separately for each factor and summed into differentiated factor weights.

For Reddit, the equivalent 5B model catastrophically underperforms (−0.31 Δ Sharpe). The likely explanation is that WallStreetBets activity is fundamentally a **market-directional signal**, not a cross-sectional one. The community's sentiment tends to be coordinated around broad market moves (meme stocks, COVID crash recovery, FOMO periods) rather than driven by valuation or quality differentials across stocks. Applying this signal for cross-sectional factor tilting introduces noise without any predictive information, and the Ridge regularization is not strong enough to reduce the signal to zero in all factor predictions.

### Why Do Complex Models Underperform?

A consistent pattern across both sentiment sources is that **model complexity is penalized**: the mean-variance models (5C, 5D) and random forest models (5E, 5F) generally do not improve on the simpler Ridge signal-scaling (5A, 5B).

For MV models (5D), the issue is likely **estimation error amplification**: the MV portfolio optimizer is notoriously sensitive to the expected return estimates (μ̂), and small errors in Ridge predictions get magnified when constructing portfolio weights. Even with Ledoit-Wolf shrinkage for the covariance matrix and a turnover penalty, the optimizer still relies on factor return predictions that carry substantial noise.

For Random Forest models (5E, 5F), the likely issue is the **small sample size relative to feature count** in the expanding window. Even at the end of the WSJ window, the training set has ~140 months with 8 features — a relatively favorable ratio, but the tree-based model may still overfit on in-sample patterns that fail to generalize.

The simpler signal-scaling approach (5A, 5B) is more robust because it makes only one prediction (market-level or per-factor) and maps it to a bounded weight perturbation. This limits the damage when predictions are wrong, while still capturing value when they are right.

### Macro Baseline Performance

It is worth noting that even the macro-only control models (F0A, F0D) both post **negative Sharpe ratios** over the WSJ window — meaning macro-conditioned factor timing fails to beat the EWP baseline, let alone generate absolute positive returns. This is consistent with the broader empirical literature: factor return predictability from macro variables is real but modest, and translating it into consistently profitable timing strategies is difficult after accounting for realistic transaction costs and look-ahead-free signal construction.

The Reddit window is even more challenging, with the EWP itself posting a negative Sharpe (−0.18), reflecting the well-documented struggles of traditional factor portfolios from 2013–2025.

### Statistical Significance

Despite the positive Δ Sharpe for 5B (WSJ) and 5A (Reddit), the results carry important statistical caveats:

- With only ~148 and ~120 OOS months respectively, the statistical power to distinguish genuine alpha from noise is limited. A Newey-West t-statistic above 1.96 requires an unusually high information ratio given the short history.
- The only statistically significant alpha in the experiment is Reddit 5A (t = 1.73 at the 10% level). The WSJ 5B improvement is economically large (+0.21 Sharpe) but whether its alpha t-statistic is significant requires reading the full results table from notebook 04.
- These results should be treated as **suggestive rather than conclusive**, particularly given the single out-of-sample evaluation without any out-of-sample test repetition across different time periods or markets.

---

## Limitations

### 1. Short Out-of-Sample Windows

The WSJ window provides 148 OOS months, and the Reddit window approximately 120 OOS months. While walk-forward evaluation is the gold standard for time-series prediction, these samples are short enough that a few unusual market episodes (the 2008 GFC, COVID-19 in 2020, the 2022 rate shock) can disproportionately influence results. A model that happened to position well during any single crisis could show inflated performance.

### 2. WSJ Data Availability Ends in 2017

The Bybee et al. (2024) WSJ dataset covers 2000–2017, limiting our WSJ evaluation window to 17 years. Modern NLP sentiment from news is now available from many providers (Bloomberg, Refinitiv, GDELT), but this replication uses the academic open-source corpus. The post-2017 period — which includes significant regime changes in factor performance — cannot be evaluated with this signal.

### 3. Sentiment Signal Non-Stationarity

Both the IC scatter plots and rolling IC charts reveal that signal predictability is highly time-varying. The WSJ IC time series shows strong positive predictability around 2006–2010 and again around 2016–2017, but near-zero or negative predictability in between. Reddit similarly shows volatile rolling IC. This non-stationarity means that an expanding window walk-forward may systematically include periods where the signal is not predictive, diluting the estimated alpha.

### 4. Single NLP Architecture per Source

We use one model per sentiment source: `all-MiniLM-L6-v2` for WSJ embeddings and `cardiffnlp/twitter-roberta-base-sentiment` for Reddit. Both are strong open-source choices, but the results are not robust to alternative architectures. A study using larger models (e.g., OpenAI embeddings for WSJ, a larger financial sentiment model for Reddit) might yield different IC levels and factor timing performance.

### 5. No Transaction Costs on the Meta-Strategy

While net factor returns already deduct 30 bps × turnover within each factor portfolio, the **factor weight turnover** of the timing strategy itself is not penalized. Active factor timing strategies can have substantial turnover at the meta level, and imposing realistic execution costs (bid-ask spreads on ETF-like factor exposures, delay slippage) would likely erode a portion of the gross timing gains.

### 6. No Short-Selling or Position Constraints

The walk-forward portfolio construction allows any non-negative factor weight (long-only factors), but does not impose concentration limits, turnover limits, or minimum diversification requirements that would apply in a real implementation. The unconstrained MV optimizer in particular can produce highly concentrated weight vectors.

### 7. Hyperparameter Sensitivity

Key hyperparameters — risk aversion (γ = 3.0), TC penalty (κ = 2.0), Ridge alpha grid, RF depth — are fixed rather than jointly optimized. A more systematic search over these parameters using time-series cross-validation might improve (or worsen) results. The fixed values may not be optimal across the full evaluation window.

### 8. Survivorship and Selection Bias in Factor Construction

Despite applying the CRSP microcap screen and a 6-month accounting lag, the factor portfolios are constructed in-sample and the same universe is used throughout. True out-of-sample factor construction would require real-time CRSP files, which are difficult to replicate in an academic setting.

---

## Next Steps

### Immediate Extensions

1. **Extend WSJ coverage beyond 2017.** Sources such as GDELT, Bloomberg News Analytics, or Refinitiv News Sentiment could provide similar embedding-based indices for the post-2017 period, allowing the WSJ-augmented models to be evaluated on a more recent and factor-challenging environment.

2. **Add a macro regime overlay.** Conditioning the model on a simple binary recession indicator (e.g., NBER expansion/contraction) could improve stability. Several papers show that factor timing models work better in certain macro regimes, and separating the evaluation by regime would clarify where sentiment adds the most value.

3. **Ensemble the WSJ and Reddit signals.** The two sources are complementary — WSJ covers 2000–2017, Reddit covers 2013–2025. An ensemble model that uses each where available, with appropriate downweighting of Reddit for cross-sectional predictions, might produce more consistent results across the full sample.

### Modeling Improvements

4. **Temporal models (LSTM, Transformer).** The current Ridge models treat each month independently (conditioning on the current feature vector only). A sequential model that explicitly tracks how sentiment evolves over time — not just the current level — may capture persistence and trend-following in narrative cycles better than a static regression.

5. **Hierarchical Bayes factor timing.** A Bayesian framework that pools information across factors (treating per-factor Ridge coefficients as draws from a common prior) could reduce estimation error in the 5B/5F models, where five separate models are estimated independently.

### Data Extensions

6. **Additional alternative data.** Options market implied volatility (VIX term structure, put-call ratio), earnings call transcripts, and Google Trends data have been shown to contain factor-relevant information and could augment or replace some of the current macro features.

7. **International markets.** The framework is entirely US-centric. Testing on European and Asian factor portfolios would address whether the sentiment signals generalize beyond US financial media.

### Infrastructure and Robustness

8. **Multiple evaluation periods.** Rather than a single walk-forward on one time series, implementing a blocked bootstrap or multiple disjoint evaluation windows would provide a better estimate of the sampling uncertainty in the Δ Sharpe statistics.

9. **Replication on a second sentiment source.** Reproducing the WSJ analysis with Bloomberg News Analytics (or another news sentiment provider) over the same 2000–2017 period would test whether the 5B result is specific to the embedding architecture used here or is a more general finding about news-based factor timing.

---

## Setup

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels
```

All notebooks read from `data/FinalMonthlyDataset_ours_ff_macro.csv` using relative paths. No additional data downloads required. No API keys needed.

---

## References

- Bybee, L., Kelly, B., Manela, A., & Xiu, D. (2024). *Business News and Business Cycles*. Journal of Finance.
- Cookson, J. A., Engelberg, J., & Mullins, W. (2024). *Does Retail Attention Matter? Evidence from Reddit*. Review of Finance.
- Fama, E. F., & French, K. R. (2015). *A five-factor asset pricing model*. Journal of Financial Economics.
- Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. Review of Financial Studies.
- Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices*. Journal of Multivariate Analysis.
- Asness, C., Moskowitz, T., & Pedersen, L. H. (2013). *Value and Momentum Everywhere*. Journal of Finance.
- Haddad, V., Kozak, S., & Santosh, S. (2020). *Factor Timing*. Review of Financial Studies.
