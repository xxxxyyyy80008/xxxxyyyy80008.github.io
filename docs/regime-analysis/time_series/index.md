---
layout: default
title: Time Series
parent: Regime Analysis
nav_order: 1
has_children: true
permalink: /docs/regime-analysis/time_series/
---

# Time Series Analysis for Finance
{: .fs-9 }

Advanced preprocessing, normalization, and forecasting techniques for non-stationary financial data.
{: .fs-6 .fw-300 }

[Adaptive Normalization](/docs/regime-analysis/time_series/adaptive-normalization){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Regime Shift Detection](/docs/regime-analysis/time_series/regime-shift){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Overview

Financial time series present unique challenges that standard machine learning preprocessing cannot address:

- **Non-stationarity** — Statistical properties change over time
- **Regime dependence** — Different market states require different models
- **Fat tails** — Extreme events occur more frequently than Gaussian assumptions
- **Volatility clustering** — Periods of high/low volatility persist
- **Look-ahead bias risk** — Future information can easily leak into training

{: .highlight }
> Traditional z-score normalization using full-sample statistics is fundamentally flawed for financial forecasting — it leaks future information and assumes stationarity.

---

## 📦 Repository

| Repository | Description |
|:-----------|:------------|
| [**time-series-research**](https://github.com/your-username/time-series-research) | Normalization methods, regime detection, forecasting models |

---

## 🎯 Key Challenges & Solutions

| Challenge | Naive Approach | Problem | Our Solution |
|:----------|:---------------|:--------|:-------------|
| **Normalization** | Global z-score | Look-ahead bias | Rolling / Adaptive normalization |
| **Non-stationarity** | Assume constant distribution | Model degradation | Regime-conditional modeling |
| **Distribution shift** | Fixed preprocessing | Poor generalization | Deep Adaptive Input Normalization |
| **Volatility clustering** | Homoscedastic models | Underestimated risk | GARCH family, regime-switching |
| **Regime changes** | Single model | Catastrophic failures | Change point detection + model switching |

---
