---
layout: default
title:  MABW Volatility Breakout & Momentum Strategy
parent: Strategies & Backtests
nav_order: 1
permalink: /systematic-strategies/strategies/mabw-volatility-breakout/
---

# Strategy: Moving Average Band Width (MABW)

**Logic:** Volatility Expansion / Trend Following
**Universe:** Liquid Large-Cap Equities
**Execution Frequency:** Daily

### **1. The Alpha Hypothesis**
Markets cycle between periods of high volatility (expansion) and low volatility (compression). The MABW strategy hypothesizes that prolonged periods of volatility compression—visualized as the "squeezing" of Bollinger Bands—often precede explosive price moves. By entering on the breakout of this compression in the direction of the dominant trend, we aim to capture the "fat tails" of the return distribution.

### **2. Mathematical Model**

The strategy relies on two primary components: Trend Identification and Volatility Compression.

#### A. Trend Filter (EMA Crossover)
We first establish the regime using an Exponential Moving Average (EMA) crossover to filter out counter-trend signals.

$$
\text{Regime} = \begin{cases} 
1 (Bullish) & \text{if } \text{EMA}_{fast} > \text{EMA}_{slow} \\
-1 (Bearish) & \text{if } \text{EMA}_{fast} < \text{EMA}_{slow}
\end{cases}
$$

#### B. Volatility Compression Signal
We utilize normalized Band Width ($BW$) to define volatility. The signal triggers only when the current bandwidth is at a local minimum (compression).

$$
BW_t = \frac{\text{UpperBand}_t - \text{LowerBand}_t}{\text{MiddleBand}_t}
$$

The entry signal is generated when the Band Width is at a historic low for the lookback window $N$:

$$
\text{Signal}_t = (\text{Regime} == 1) \land (BW_t \approx \min(BW_{t-N} \dots BW_t))
$$

### **3. Implementation Details**
To ensure performance during optimization, the signal generation is fully vectorized using Pandas.

```python
def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
    # 1. Calculate EMAs for Trend
    data['ema_fast'] = data['close'].ewm(span=self.params['ema_fast_span']).mean()
    data['ema_slow'] = data['close'].ewm(span=self.params['ema_slow_span']).mean()
    
    # 2. Calculate Band Width
    rolling_std = data['close'].rolling(window=self.params['bb_window']).std()
    data['bw'] = (4 * rolling_std) / data['close'] # Simplified calculation
    
    # 3. Identify Compression (Lowest Width in N days)
    # This vectorization avoids a slow Python for-loop
    data['min_bw'] = data['bw'].rolling(window=self.params['lookback_period']).min()
    data['is_compressed'] = np.isclose(data['bw'], data['min_bw'], atol=1e-5)
    
    # 4. Generate Entry Signal
    data['long_entry'] = (data['ema_fast'] > data['ema_slow']) & data['is_compressed']
    
    return data
```

### **4. Walk-Forward Performance (Results)**

The strategy was validated using a Walk-Forward Analysis with a training window of 252 days and a testing window of 63 days.

| Metric | In-Sample (Mean) | Out-of-Sample (Mean) | Delta |
| :--- | :--- | :--- | :--- |
| **Sharpe Ratio** | 2.15 | **1.42** | -34% |
| **Win Rate** | 58.2% | **55.1%** | -3.1% |
| **Max Drawdown** | -12.5% | **-18.4%** | +5.9% |

**Commentary:**
As expected, the Sharpe ratio degrades in the Out-of-Sample (OOS) data. However, the **Win Rate** remains remarkably stable (only a 3% drop), indicating that the mechanical edge (volatility expansion) is robust, even if the magnitude of the returns (Profit Factor) was slightly overfitted in training.

### **5. Parameter Stability**
Running the stability analysis on the `lookback_period` parameter (used to determine the compression window) revealed a Coefficient of Variation (CV) of **0.22**.

*   **Interpretation:** The optimizer consistently preferred a lookback period between 18 and 22 days across different market years. This stability suggests the "20-day cycle" is a persistent market feature, rather than a transient anomaly.

### **6. Defect Analysis & Future Improvements**
*Why does this strategy fail?*

1.  **The "Head-Fake" Breakout:** The primary source of losses is entering a breakout that immediately reverses (a "bull trap").
    *   *Proposed Fix:* Implement a **Volume Confirmation** filter. Require volume on the breakout day to be $> 150\%$ of the 10-day average.
2.  **Choppy Regimes:** During sideways markets with high noise but no clear trend, the EMA filter whipsaws, and volatility compression signals trigger frequently with no follow-through.
    *   *Proposed Fix:* Add an ADX (Average Directional Index) filter to ensure the market is trending before enabling the system.

