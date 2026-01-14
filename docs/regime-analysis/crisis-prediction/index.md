---
layout: default
title: Crisis Prediction
parent: Market Regime Analysis
nav_order: 2
has_toc: false
has_children: true
---

# Crisis & Crash Prediction Models
{: .fs-7 }

Machine learning approaches to market crash detection and early warning systems.
{: .fs-5 .fw-300 }


# Crisis Prediction

This section documents the **crisis regime prediction** component of the project: how we define “crisis”, build a model to estimate **forward-looking crash probability**, and evaluate it under **time-series constraints**.

The core objective is to produce a daily signal:

$$
p_{t,H} = \mathbb{P}(y_{t+H}=1 \mid X_t)
$$

where $$H$$ is the prediction horizon (e.g., 1/3/5/10 trading days).

---

## What “crisis” means here

We model equity markets as having two regimes:

- **Normal regime (0):** typical market conditions; equity risk premium dominates.
- **Crisis regime (1):** stress/crash conditions; risk management actions may be warranted.

### Target definition (implemented)
Using S&P 500 (Yahoo `^GSPC`), we define a crisis indicator from the 15-day return:

$$
r^{(15)}_t = \frac{P_t}{P_{t-15}} - 1
$$

Within a long rolling window, we flag crisis when the latest 15-day return falls below the rolling 5th percentile:

$$
y_t = \mathbb{1}\Big[r^{(15)}_t < q_{0.05}\big(r^{(15)}_{t-2150:t-1}\big)\Big]
$$

We then build **forward targets** by shifting the crisis indicator:

- `target_1d`, `target_3d`, `target_5d`, `target_10d`  
- Default modeling target: **`target = target_3d`**

---

## Data sources (high level)

### Market data (Yahoo Finance)
Daily adjusted market series across:
- US equities and volatility (S&P 500, Nasdaq, VIX, etc.)
- US yields (2Y/5Y/10Y proxies)
- FX proxies (USD index, JPY)
- Global equities (EU/Asia indices, EM ETFs)
- Commodities (Gold, WTI oil)

### Macro / financial conditions (FRED)
Daily-aligned macro/financial indicators including:
- Financial stress / conditions (e.g., NFCI, STLFSI)
- Rates and spreads (e.g., term spreads, funding spreads)
- Growth/inflation proxies and USD indices
- Energy price series (e.g., WTI)

> Lower-frequency FRED series are forward-filled to daily frequency for alignment.

---

## Features (high level)

We generate a structured feature set designed to capture:

- **Trend & momentum:** multi-horizon changes, ROC, moving-average distances
- **Risk & uncertainty:** rolling volatility proxies, VIX-related interactions
- **Normalization:** rolling z-scores
- **Risk-adjusted behavior:** rolling Sharpe-style ratios
- **Cross-asset relationships:** curated interaction features (e.g., equity vs VIX, equity vs yields, gold/oil ratio)

Examples of engineered feature families:
- `*_pct_chg{5,10,20,60,120,250}`
- `*_std{60,125}`, `*_volat{60,125}`
- `*_zscore{60,120,200,250}`
- `*_sharpe{120,250}`
- Interactions like `{market}_div_VIX`, `Gold_Oil_Ratio`, `SP500_div_NFCI`

---

## Modeling approach

We frame crisis prediction as **binary classification with severe class imbalance**:

- Primary model family: **GBDT** (LightGBM / XGBoost)
- Outputs: calibrated probabilities $$p_{t,H}$$ and threshold-based alerts

Key ideas:
- Avoid leakage via strict time-based splits
- Prefer probability quality (calibration) over accuracy
- Evaluate operational trade-offs (false alarms vs missed crises)

---

## Evaluation (time-series first)

We use **walk-forward / expanding window** validation rather than random splits to reflect real-world deployment.

Common metrics:
- PR-AUC (robust under imbalance)
- ROC-AUC (supplementary)
- Brier score (probability quality)
- Recall at fixed false-positive rate (alerting usability)

---

## Outputs

Typical artifacts produced by this section:

- A model-ready dataset (Parquet)
- Trained model + feature list + config snapshot
- Daily crisis probability time series
- Evaluation report (metrics + plots)

---

## How to navigate this section

- **Data Prep:** how raw data is downloaded, aligned, and saved
- **Feature Engineering:** feature families and interactions
- **Training:** model configs, hyperparameters, class weighting
- **Interpretation:** feature importance / diagnostics
- **Deployment:** end-to-end scoring pipeline and monitoring


