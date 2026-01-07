---
layout: default
title: Technical Indicators
parent: Alpha Research & Signal Generation
nav_order: 1
has_children: true
permalink: /docs/alpha-research/indicators/
---

# Technical Indicators Library
{: .fs-7 }

A collection of over 50 technical indicators implemented in Python for quantitative analysis and trading strategies.
{: .fs-5 .fw-300 }

[Trend Indicators](/docs/alpha-research/indicators/trend/){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Momentum Indicators](/docs/alpha-research/indicators/momentum/){: .btn .fs-5 .mb-4 .mb-md-0 }
[Volatility Indicators](/docs/alpha-research/indicators/volatility/){: .btn .fs-5 .mb-4 .mb-md-0 }
[Volume Indicators](/docs/alpha-research/indicators/volume/){: .btn .fs-5 .mb-4 .mb-md-0 }
[Oscillators](/docs/alpha-research/indicators/oscillators/){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Overview

This section provides a comprehensive library of technical indicators categorized by their function. Each indicator includes:

- **Mathematical Definitions** — Detailed formulas and explanations
- **Python Implementations** — Ready-to-use code snippets
- **Use Cases** — Examples of how to apply each indicator in trading strategies

---

## 📦 Repository

| Repository | Description |
|:-----------|:------------|
| [**technical-indicators**](https://github.com/your-username/technical-indicators) | Complete library of indicators with implementations and documentation |

---

## 📊 Indicator Categories

### Trend Indicators
- **Simple Moving Average (SMA)**
- **Exponential Moving Average (EMA)**
- **Weighted Moving Average (WMA)**
- **Fractal Adaptive Moving Average (FRAMA)**
- **Moving Average Convergence Divergence (MACD)**

### Momentum Indicators
- **Relative Strength Index (RSI)**
- **Stochastic Oscillator**
- **Rate of Change (ROC)**
- **Commodity Channel Index (CCI)**
- **Momentum Indicator**

### Volatility Indicators
- **Average True Range (ATR)**
- **Bollinger Bands**
- **Keltner Channel**
- **Donchian Channel**
- **Historical Volatility**

### Volume Indicators
- **On-Balance Volume (OBV)**
- **Accumulation/Distribution Line**
- **Chaikin Money Flow**
- **Volume Rate of Change**
- **Money Flow Index (MFI)**

### Oscillators
- **Stochastic RSI**
- **Williams %R**
- **Ultimate Oscillator**
- **Chaikin Oscillator**
- **Price Oscillator**

---

## 🔍 Featured Indicator Implementations

### Simple Moving Average (SMA)
#### Definition
The Simple Moving Average (SMA) is calculated as the average of a specified number of past prices.

$$
SMA_t = \frac{1}{N} \sum_{i=0}^{N-1} P_{t-i}
$$

Where:
- $$P_t$$ = price at time $$t$$
- $$N$$ = number of periods

#### Python Implementation

```python
import pandas as pd

def simple_moving_average(prices, window):
    return prices.rolling(window=window).mean()
```

#### Use Case
Use SMA to identify the general direction of the trend. A price crossing above the SMA may indicate a bullish trend, while crossing below may indicate a bearish trend.

---

### Relative Strength Index (RSI)
#### Definition
The Relative Strength Index (RSI) measures the speed and change of price movements.

$$
RSI = 100 - \frac{100}{1 + RS}
$$

Where:
- $$RS = \frac{\text{Average Gain}}{\text{Average Loss}}$$

#### Python Implementation

```python
def relative_strength_index(prices, window):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

#### Use Case
RSI is typically used to identify overbought or oversold conditions in a market. An RSI above 70 indicates overbought conditions, while below 30 indicates oversold conditions.

---

### Average True Range (ATR)
#### Definition
The Average True Range (ATR) measures market volatility by decomposing the entire range of an asset for that period.

$$
ATR = \frac{1}{N} \sum_{i=1}^{N} TR_i
$$

Where:
- $$TR = \max(\text{High} - \text{Low}, |\text{High} - \text{Close}_{previous}|, |\text{Low} - \text{Close}_{previous}|)$$

#### Python Implementation

```python
def average_true_range(df, window):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()
```

#### Use Case
ATR is used to gauge market volatility; higher ATR values indicate higher volatility.

---

## 📚 References

- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*
- Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*
- Achelis, S. B. (2001). *Technical Analysis from A to Z*

---

## 🔗 Related Sections

- [Alpha Research](/docs/alpha-research/) — Explore trading strategies utilizing these indicators.
- [Backtesting Framework](/docs/alpha-research/strategies/backtest-framework/) — Test your strategies with these indicators effectively.

---

{: .warning }
> **Disclaimer**: The implementations provided here are for educational purposes only. Ensure to backtest any strategy before applying it in real trading.
```

---

This index page:

- ✅ Categorizes indicators by function (trend, momentum, volatility, volume, oscillators)
- ✅ Provides detailed definitions, Python implementations, and use cases for selected indicators
- ✅ Links to relevant child pages for further exploration of specific categories
- ✅ Includes a **repository** link for easy access to the codebase
- ✅ Contains **references** for further reading and verification

Would you like me to help create a specific child page for any category, such as **trend indicators** or **momentum indicators**?