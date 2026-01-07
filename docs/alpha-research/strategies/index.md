---
layout: default
title: Trading Strategies
parent: Alpha Research
nav_order: 2
has_children: true
permalink: /docs/alpha-research/strategies/
---

# Trading Strategies & Backtesting
{: .fs-9 }

Systematic rule-based strategies, backtesting framework, and rigorous performance evaluation.
{: .fs-6 .fw-300 }

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

## 📈 Performance Overview

### Out-of-Sample Results (2020–2025)

| Strategy | CAGR | Sharpe | Sortino | Max DD | Calmar | Win Rate | Profit Factor |
|:---------|:----:|:------:|:-------:|:------:|:------:|:--------:|:-------------:|
| FRAMA Crossover | 12.3% | 1.42 | 1.98 | -15.2% | 0.81 | 54% | 1.62 |
| RSI Mean Reversion | 9.8% | 1.18 | 1.55 | -18.7% | 0.52 | 61% | 1.48 |
| IFT RSI Momentum | 11.5% | 1.35 | 1.82 | -16.8% | 0.68 | 52% | 1.55 |
| Bollinger Breakout | 8.1% | 0.95 | 1.21 | -22.4% | 0.36 | 43% | 1.38 |
| Keltner Squeeze | 10.2% | 1.22 | 1.64 | -17.5% | 0.58 | 48% | 1.51 |
| Multi-Factor Ensemble | 14.5% | 1.67 | 2.24 | -12.8% | 1.13 | 52% | 1.78 |
| Regime-Adaptive | **15.8%** | **1.82** | **2.45** | **-11.2%** | **1.41** | 55% | **1.92** |
| *Benchmark (SPY B&H)* | *10.1%* | *0.72* | *0.89* | *-33.7%* | *0.30* | *—* | *—* |

{: .note }
All results include 10bps transaction costs, 5bps slippage, and 1-day execution delay. Risk-free rate assumed 4%.

---

