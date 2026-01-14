---
layout: default
title: Crisis Prediction
parent: Market Regime Analysis
nav_order: 2
has_toc: false
has_children: true
permalink: /docs/regime-analysis/crisis-prediction
---

# Crisis & Crash Prediction Models
{: .fs-7 }

Machine learning approaches to market crash detection and early warning systems.
{: .fs-5 .fw-300 }


## Definition of "Crisis"

Equity markets are modeled as having two regimes:

- **Normal regime (0):** Typical market conditions, where the equity risk premium is dominant.
- **Crisis regime (1):** Stress or crash conditions, where risk management actions may be warranted.

The core objective is to produce a daily signal:

$$
p_{t,H} = \mathbb{P}(y_{t+H}=1 \mid X_t)
$$

where $$H$$ is the prediction horizon (e.g., 1/3/5/10 trading days).


### Target Definition 
Using S&P 500 data (Yahoo `^GSPC`), a crisis indicator is defined from the 15-day return:

$$
r^{(15)}_t = \frac{P_t}{P_{t-15}} - 1
$$

Within a long rolling window, a crisis is flagged when the latest 15-day return falls below the rolling 5th percentile:

$$
y_t = \mathbb{1}\Big[r^{(15)}_t < q_{0.05}\big(r^{(15)}_{t-2150:t-1}\big)\Big]
$$

**Forward targets** are then constructed by shifting the crisis indicator:
- `target_1d`, `target_3d`, `target_5d`, `target_10d`
- The default modeling target is **`target = target_3d`**.

---

## Data Sources

### Market Data (Yahoo Finance)
Daily adjusted market series are used across the following categories:
- US equities and volatility (S&P 500, Nasdaq, VIX)
- US yields (2Y/5Y/10Y proxies)
- FX proxies (USD index, JPY)
- Global equities (EU/Asia indices, EM ETFs)
- Commodities (Gold, WTI oil)

### Macro / Financial Conditions (FRED)
Daily-aligned indicators are included, such as:
- Financial stress/conditions indices (e.g., NFCI, STLFSI)
- Rates and spreads (e.g., term spreads, funding spreads)
- Growth/inflation proxies and USD indices
- Energy price series (e.g., WTI)

> Lower-frequency FRED series are forward-filled to daily frequency for alignment.

---

## Features

A structured feature set is generated to capture the following:
- **Trend & momentum:** Multi-horizon changes, rate-of-change, moving-average distances
- **Risk & uncertainty:** Rolling volatility proxies, VIX-related interactions
- **Normalization:** Rolling z-scores
- **Risk-adjusted behavior:** Rolling Sharpe-style ratios
- **Cross-asset relationships:** Curated interaction features (e.g., equity vs VIX, equity vs yields, gold/oil ratio)

Examples of engineered feature families include:
- `*_pct_chg{5,10,20,60,120,250}`
- `*_std{60,125}`, `*_volat{60,125}`
- `*_zscore{60,120,200,250}`
- Interactions such as `{market}_div_VIX`, `Gold_Oil_Ratio`, `SP500_div_NFCI`

---

## Modeling Approach

Crisis prediction is framed as **binary classification with severe class imbalance**.
- Primary model family: **GBDT** (LightGBM / XGBoost)
- Outputs: Calibrated probabilities $$p_{t,H}$$ and threshold-based alerts

Modeling priorities:
- Leakage is avoided through strict time-based splits.
- Probability quality (calibration) is prioritized over accuracy.
- Operational trade-offs (false alarms vs missed crises) are evaluated.

---

## Evaluation (Time-Series)

**Walk-forward / expanding window** validation is used instead of random splits to reflect real-world deployment.

Common metrics include:
- PR-AUC (selected for robustness under imbalance)
- ROC-AUC (supplementary)
- Brier score (for probability quality)
- Recall at a fixed false-positive rate (for alerting usability)

---

## Outputs

The following artifacts are produced:
- A model-ready dataset (Parquet format)
- A trained model, feature list, and configuration snapshot
- A daily time series of crisis probabilities
- An evaluation report (metrics and plots)

