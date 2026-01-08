---
layout: default
title: Strategies & Backtests
parent: Alpha Research & Signal Generation
nav_order: 2
has_children: true
has_toc: false
permalink: /docs/alpha-research/strategies/
---

# Trading Strategies & Backtesting
{: .fs-7 }

Systematic rule-based strategies, backtesting, and performance evaluation.
{: .fs-5 .fw-300 }

[Backtest Framework](/docs/alpha-research/strategies/backtest-framework){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Strategy Results](/docs/alpha-research/strategies/results){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Overview

This section documents systematic trading strategies built using the [technical indicators library](/docs/alpha-research/indicators/), along with the backtesting infrastructure used to evaluate them:

- **Rule-Based Strategies** — Fully systematic, no discretionary decisions
- **Vectorized Backtesting** — Fast historical simulation engine
- **Realistic Assumptions** — Transaction costs, slippage, execution delays
- **Rigorous Evaluation** — Walk-forward testing, robustness checks

{: .highlight }
> All strategies are designed for research and educational purposes. Past performance does not guarantee future results.

---

## 📦 Repositories

| Repository | Description |
|:-----------|:------------|
| [**trading-strategies**](https://github.com/your-username/trading-strategies) | Strategy implementations and backtest code |
| [**backtest-engine**](https://github.com/your-username/backtest-engine) | Core backtesting framework |

---

## 📊 Strategy Library

### Strategy Summary

| Strategy | Type | Indicators | Timeframe | Complexity |
|:---------|:-----|:-----------|:----------|:-----------|
| [FRAMA Crossover](#frama-crossover) | Trend | FRAMA, ATR | Daily | Low |
| [RSI Mean Reversion](#rsi-mean-reversion) | Mean Reversion | RSI, Bollinger | Daily | Low |
| [IFT RSI Momentum](#ift-rsi-momentum) | Momentum | IFT RSI | Daily | Low |
| [Bollinger Breakout](#bollinger-breakout) | Breakout | BB, ATR | Daily | Medium |
| [Keltner Squeeze](#keltner-squeeze) | Volatility | KC, BB, MACD | Daily | Medium |
| [Multi-Factor Ensemble](#multi-factor-ensemble) | Ensemble | Multiple | Daily | High |
| [Regime-Adaptive](#regime-adaptive-strategy) | Adaptive | HMM + Multiple | Daily | High |

---

