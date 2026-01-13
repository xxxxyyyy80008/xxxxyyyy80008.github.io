---
layout: default
title: Time Series Modeling
parent: Market Regime Analysis
nav_order: 4
has_children: true
has_toc: false
permalink: /docs/regime-analysis/time-series/
---

# Time Series Modeling
{: .fs-7 }

Two quant-native lenses on market state: **latent regimes** (discrete, persistent) and **anomaly scores** (continuous, stress/surprise).

{: .fs-5 .fw-300 }

[Markov-Switching Regime Detection](/docs/regime-analysis/time_series/markov-switching){: .btn .btn-primary .fs-3 .mb-4 .mb-md-0 .mr-2 }
[Anomaly Detection](/docs/regime-analysis/time-series/anomaly-detection){: .btn .fs-3 .mb-4 .mb-md-0 }

---


## 1) Markov-Switching Regime Detection

**Goal:** infer a hidden regime $$S_t$$ where both conditional mean dynamics and dispersion can change.

**Label (research proxy):** 20-day forward max return
$$
y_t = 100 \times \frac{\max(C_{t+1},\dots,C_{t+20}) - C_t}{C_t}
$$

**Best model (selected by BIC on train):** 3 regimes, switching variance, exog `lag1` + `lag20`
$$
y_t=\beta_{0,S_t}+\beta_{1,S_t}y_{t-1}+\beta_{2,S_t}y_{t-20}+\varepsilon_t,\quad
\varepsilon_t\sim\mathcal{N}(0,\sigma^2_{S_t})
$$

**Key results (train, 2000-02-01→2025-07-22; $$n=6406$$):**
- Model selection: `3_regime_full` BIC **18433.66** (wins vs 2-regime, 5-regime, and non-switching-variance variants)
- Regime mix + behavior (hard assignment via $$\arg\max_i P(S_t=i)$$):
  - **R0 (53.5%)**: mean **2.248%**, std **1.572%**, longest **343** days
  - **R1 (5.0%)**: mean **6.948%**, std **5.925%**, longest **74** days *(high opportunity + high dispersion)*
  - **R2 (41.5%)**: mean **3.322%**, std **2.837%**, longest **131** days
- Persistence (expected duration $$\mathbb{E}[D_i]=1/(1-p_{ii})$$):
  - R0 **40.6d**, R1 **16.1d**, R2 **25.2d**
- Transition matrix is diagonal-dominant (persistent regimes; rare regime is short-lived).


---

## 2) Anomaly Detection

**Goal:** produce a continuous anomaly score $$a_t$$ capturing “unusualness” (stress, structural breaks, outliers) without labeled anomalies.

**Detectors used:** Isolation Forest, VAE, and an ensemble.

**Key takeaway:**
- **VAE achieved the strongest correlation** with the forward-return proxy on both train and test.
- VAE scores are typically **larger in magnitude** → treat score scale as *model-specific calibration*, not “more anomalous reality”.


---
