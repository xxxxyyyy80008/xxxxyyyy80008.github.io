---
layout: default
title: Alpha Research & Signal Generation
nav_order: 5
has_children: true
permalink: /docs/alpha-research/
---

# Alpha Research & Signal Generation
{: .fs-9 }

Technical indicator implementations, trading strategy development, and systematic backtesting frameworks.
{: .fs-6 .fw-300 }

[View Indicators](/docs/alpha-research/indicators/){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View Strategies](/docs/alpha-research/strategies/){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Overview

This section contains my research on quantitative alpha generation, featuring:

- **50+ Technical Indicators** — Custom implementations from scratch in Python
- **Rule-Based Strategies** — Systematic trading strategies using indicator signals
- **Backtesting Framework** — Performance evaluation with realistic assumptions

{: .highlight }
> All indicators are implemented from mathematical definitions without relying on TA-Lib, ensuring full understanding of the underlying calculations.

---

## 📦 Repositories

| Repository | Description |
|:-----------|:------------|
| [**technical-indicators**](https://github.com/your-username/technical-indicators) | 50+ indicator implementations with documentation |
| [**trading-strategies**](https://github.com/your-username/trading-strategies) | Strategy implementations and backtest results |

---

## 📊 Indicator Library Summary

### By Category

| Category | Count | Examples |
|:---------|:-----:|:---------|
| **Trend** | 12 | SMA, EMA, DEMA, TEMA, KAMA, FRAMA, T3 |
| **Momentum** | 15 | RSI, IFT RSI, Stochastic, Williams %R, CCI, CMO |
| **Volatility** | 10 | ATR, Bollinger Bands, Keltner Channel, Donchian |
| **Volume** | 8 | OBV, VWAP, MFI, Chaikin Money Flow, A/D Line |
| **Oscillators** | 7 | MACD, PPO, APO, TRIX, Ultimate Oscillator |

### Featured Implementations

<div class="code-example" markdown="1">

#### FRAMA (Fractal Adaptive Moving Average)
{: .fs-5 .fw-500 }

Adapts smoothing based on fractal dimension of price series — more responsive in trends, smoother in ranges.

$$
\alpha = \exp(-4.6 \cdot (D - 1))
$$

Where $$D$$ is the fractal dimension calculated from price highs and lows.

[View Implementation →](/docs/alpha-research/indicators/trend#frama)
{: .fs-4 }

</div>

<div class="code-example" markdown="1">

#### IFT RSI (Inverse Fisher Transform RSI)
{: .fs-5 .fw-500 }

Applies inverse Fisher transform to RSI for sharper turning point signals with values bounded $$[-1, +1]$$.

$$
\text{IFT} = \frac{e^{2x} - 1}{e^{2x} + 1}
$$

Where $$x$$ is the smoothed, normalized RSI.

[View Implementation →](/docs/alpha-research/indicators/oscillators#ift-rsi)
{: .fs-4 }

</div>

---

## 📈 Strategy Performance Summary

| Strategy | Sharpe | CAGR | Max DD | Win Rate |
|:---------|:------:|:----:|:------:|:--------:|
| FRAMA Crossover | 1.42 | 12.3% | -15.2% | 54% |
| RSI Mean Reversion | 1.18 | 9.8% | -18.7% | 61% |
| Bollinger Breakout | 0.95 | 8.1% | -22.4% | 43% |
| Multi-Factor Combo | 1.67 | 14.5% | -12.8% | 52% |

{: .note }
All results are out-of-sample (2020-2025) with 10bps transaction costs and 1-day execution delay.

---

## 🔬 Research Approach

```mermaid
flowchart LR
    A[Market Data] --> B[Feature Engineering]
    B --> C[Indicator Calculation]
    C --> D[Signal Generation]
    D --> E[Strategy Rules]
    E --> F[Backtest Engine]
    F --> G[Performance Analysis]
    G --> H{Robust?}
    H -->|Yes| I[Paper Trading]
    H -->|No| B
