---
layout: default
title: Market Regime Analysis
nav_order: 4
has_children: true
has_toc: false
permalink: /docs/regime-analysis/
---

# Market Regime Analysis
{: .fs-7 }

Time series modeling, regime shift detection, equity market crisis prediction, and historical market pattern analysis.
{: .fs-5 .fw-300 }

[Equity Market Crisis Models](/docs/regime-analysis/crisis-prediction/){: .btn .btn-primary .fs-3 .mb-4 .mb-md-0 .mr-2 }
[Historical Market Pattern Analysis: 1998 vs 2025](/docs/regime-analysis/pattern-analysis){: .btn .fs-3 .mb-4 .mb-md-0 }
[Time Series Modeling](/docs/regime-analysis/time-series){: .btn .fs-3 .mb-4 .mb-md-0 }

---

## Overview

Understanding market regimes is critical for risk management and strategy adaptation. Market regimes fundamentally alter the statistical properties of returns. Strategies that ignore regime changes often fail catastrophically during transitions. This section covers:

- [**Equity Market Crisis Prediction**](/docs/regime-analysis/crisis-prediction/) — Machine learning models for crash early warning
- **Historical Market Pattern Analysis** — Comparative study of market recovery patterns
- **Regime Shift Detection** — Identifying structural breaks in market behavior

| Question | Approach | Section |
|:---------|:---------|:--------|
| How to normalize non-stationary financial series? | Deep Adaptive Input Normalization | [Time Series](/docs/regime-analysis/time-series/) |
| When do market regimes change? | Hidden Markov Models, Change Point Detection | [Regime Shift](/docs/regime-analysis/time-series/regime-shift) |
| Can we predict market crashes? | Classification ML with macro/technical features | [Crisis Prediction](/docs/regime-analysis/crisis-prediction/) |
| How do recoveries compare across eras? | Pattern matching, drawdown analysis | [Historical](/docs/regime-analysis/historical/) |



---

##  Repositories

| Repository | Description |
|:-----------|:------------|
| [**Equity-Market-Crisis-Prediction-Models**](https://github.com/xxxxyyyy80008/Equity-Market-Crisis-Prediction-Models/tree/main/notebooks){:target="_blank" rel="noopener noreferrer"}  | ML models for market crash detection |
| [**Market-Recovery-Pattern-1998-vs-2025**](https://github.com/xxxxyyyy80008/Market-Recovery-Pattern-1998-vs-2025){:target="_blank" rel="noopener noreferrer"}  | 1998 vs 2025 comparative study |
| [**Time Series Modeling**](https://github.com/xxxxyyyy80008/time-series-analysis/tree/main/notebooks){:target="_blank" rel="noopener noreferrer"}  |  latent regimes (discrete, persistent) and anomaly scores (continuous, stress/surprise) |





