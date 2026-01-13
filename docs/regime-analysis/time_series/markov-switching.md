---
layout: default
title: "Markov-Switching Regime Detection"
parent: Market Regime Analysis
grand_parent: Time Series Modeling
nav_order: 2
---

# Markov-Switching Regime Detection (S&P 500) — Statsmodels `MarkovRegression`

This project fits **Markov-switching dynamic regression** models to detect **latent regimes** in a market-derived target series. We compare multiple specifications (different number of regimes, lag features, switching variance), select the best model by **BIC**, and analyze regime persistence, transition dynamics, and regime-conditional behavior.

---

## Data & target construction

- Universe: S&P 500 index (`^GSPC`)
- Source: `yfinance`
- Sample downloaded: **6546** daily records from **2000-01-03** to **2026-01-12**
- Forward window: **20** trading days

### Forward 20-day max return target
For each date $$t$$ with close $$C_t$$:

$$
target_t = 100 \times \frac{\max(C_{t+1}, \dots, C_{t+20}) - C_t}{C_t}
$$

**Observed target statistics (after dropping tail NaNs):**
- Mean: **2.92%**
- Std: **2.73%**
- Range: **[-5.27%, 28.48%]**

### Lagged features
We build autoregressive inputs from the target:
- `lag1 = target_{t-1}`
- `lag20 = target_{t-20}`

Final modeling dataset: **6506** observations.

### Train/test split (time-based)
- Train: **6406** samples (**2000-02-01 → 2025-07-22**)
- Test: **100** samples (**2025-07-23 → 2025-12-11**)

---

## Model zoo (what we fit)

All models are estimated using `statsmodels.tsa.MarkovRegression` on the **training** set:

1. `2_regime`: 2 regimes, switching intercept  
2. `5_regime`: 5 regimes, switching intercept  
3. `3_regime_exog`: 3 regimes, switching intercept + exog (`lag1`, `lag20`)  
4. `3_regime_var`: 3 regimes, switching variance + exog (`lag20`)  
5. `3_regime_full`: 3 regimes, switching variance + exog (`lag1`, `lag20`)  

---

## Model comparison (actual results)

Lower AIC/BIC indicates better fit (with BIC penalizing complexity more strongly).

| Model | Description | K | Params | LogLik | AIC | BIC |
|---|---|---:|---:|---:|---:|---:|
| `3_regime_full` | 3 regimes, switching variance, lag1 & lag20 | 3 | 18 | -9137.95 | **18311.89** | **18433.66** |
| `3_regime_exog` | 3 regimes, switching intercept & exog (lag1, lag20) | 3 | 16 | -10240.30 | 20512.60 | 20620.84 |
| `5_regime` | 5 regimes, switching intercept | 5 | 26 | -11358.68 | 22769.37 | 22945.26 |
| `3_regime_var` | 3 regimes, switching variance, lag20 | 3 | 15 | -11771.77 | 23573.55 | 23675.02 |
| `2_regime` | 2 regimes, switching intercept | 2 | 5 | -13831.51 | 27673.02 | 27706.85 |

**Selected best model (lowest BIC):** `3_regime_full`

---

## Best model: `3_regime_full` (what it learned)

**Specification**
- 3 regimes
- switching variance enabled
- exogenous regressors: `lag1`, `lag20`
- trend/intercept included

### Regime persistence (expected durations)
- Regime 0: **40.55 days**
- Regime 1: **16.11 days**
- Regime 2: **25.15 days**

### Transition probability matrix (smoothed estimates)

| From \ To | 0 | 1 | 2 |
|---:|---:|---:|---:|
| **0** | 0.9753 | 0.0003 | 0.0315 |
| **1** | 0.0000 | 0.9379 | 0.0083 |
| **2** | 0.0247 | 0.0617 | 0.9602 |

Interpretation (high level):
- Regime 0 and Regime 2 are **highly persistent** (diagonal near 0.96–0.98).
- Regime 1 is **rarer and shorter-lived**, with persistence ~0.94 and more frequent transitions out.

---

## Regime period statistics (training set)

Regimes are assigned via $$\arg\max_i P(S_t=i)$$ using smoothed marginal probabilities.

### Regime 0 (dominant, “low dispersion”)
- Frequency: **3428 days (53.5%)**
- Mean target: **2.248%**
- Std: **1.572%**
- Min/Max: **-1.706% / 9.420%**
- Longest streak: **343** consecutive days

### Regime 1 (rare, “high opportunity / high dispersion”)
- Frequency: **320 days (5.0%)**
- Mean target: **6.948%**
- Std: **5.925%**
- Min/Max: **-5.268% / 28.478%**
- Longest streak: **74** consecutive days

### Regime 2 (common, “medium dispersion”)
- Frequency: **2658 days (41.5%)**
- Mean target: **3.322%**
- Std: **2.837%**
- Min/Max: **-3.513% / 14.941%**
- Longest streak: **131** consecutive days

---


## Notes / caveats
- This pipeline fits regimes on a **forward-looking** target (20-day forward max return). That’s useful for analysis, but for *real-time* regime detection you’d typically fit regimes on **observable features** (returns, realized vol, breadth, macro proxies).
- Score/regime labels are **model-dependent**: “Regime 1” here means the regime with the estimated characteristics above (rare + high dispersion), not a universal mapping.
- The best model was selected by **in-sample BIC** on the training period; if you plan to deploy regimes, add out-of-sample validation (forecast utility, stability across windows, rolling refits).

---
