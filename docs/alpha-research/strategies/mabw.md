---
layout: default
title:  "MABW Volatility Breakout & Momentum Strategy"
parent: Strategies
nav_order: 1
permalink: /docs/alpha-research/strategies/mabw/
---

# MABW: Volatility Breakout Strategy
{: .fs-7 }

A systematic trend-following strategy that exploits volatility clustering by entering trends during periods of extreme compression and exiting during excessive expansion.
{: .fs-5 .fw-300 }

---

## Strategy Abstract

**Logic Class:** Volatility Expansion / Breakout  
**Timeframe:** Daily  
**Universe:** Large-cap Equities & Indices (QQQ, SPY, NVDA, etc.)

The MABW strategy is predicated on the **Volatility Clustering** hypothesis (Mandelbrot, 1963), which posits that large changes in asset prices tend to be followed by large changes, and small changes by small changes. This strategy identifies "Squeeze" regimes—periods where the spread between fast and slow moving averages is at a historical low—and enters when momentum (EMA) breaks out of this compression zone.

---

## Signal Logic

The strategy utilizes a functional, state-free logic flow to generate signals.

### 1. Indicator Construction
*   **MABW Bands:** Constructed from the spread between a Fast MA (10) and Slow MA (60).
    *   $$ \text{Width} = \text{MA}_{fast} - \text{MA}_{slow} $$ (Normalized)
*   **Regime Filter (LLV):** The Lowest Low Value of the Width over a lookback period (10 days).
*   **Signal Line:** An EMA (20) of the Close price.

### 2. Entry Signal (Long Only)
A buy signal is generated if and only if **both** conditions are met simultaneously:
1.  **Compression:** The current `MAB_WIDTH` is equal to its $$N$$-day low ($$ \text{Width}_t \approx \text{LLV}_N $$).
2.  **Breakout:** The `EMA` crosses *above* the `MAB_UPPER` band.

$$
\text{Signal}_{Entry} = (\text{EMA}_t > \text{Upper}_t) \land (\text{Width}_t \le \text{LLV}_{10})
$$

### 3. Exit Signal
The trade is closed when volatility expands beyond a sustainable threshold, indicating potential trend exhaustion or reversal risk.

$$
\text{Signal}_{Exit} = \text{Width}_t > \text{Critical Threshold}_{30}
$$

---

## Implementation Details

### Core Logic Snippet
The logic is fully vectorized using pure boolean series operations.

```python
def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
    """
    Entry: EMA crosses above MAB_UPPER + MAB_WIDTH is at LLV
    Exit: MAB_WIDTH crosses above critical level
    """
    # 1. Detect Volatility Squeeze
    # Check if current width is effectively at the N-day low
    is_squeeze = (data['MAB_WIDTH'] <= data['MAB_LLV'] + 1e-6)

    # 2. Detect Momentum Breakout
    # EMA(20) crossing above the Upper MABW Band
    is_breakout = detect_crossover_above(data, 'EMA', 'MAB_UPPER')

    # Combine for Entry
    entry_cond = is_breakout & is_squeeze
    
    # 3. Detect Volatility Blow-off (Exit)
    exit_cond = detect_threshold_cross_above(data, 'MAB_WIDTH', self.mab_width_critical)
    
    return self._compile_signals(data, entry_cond, exit_cond)
```

### Configuration Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `fast_period` | 10 | Lookback for the fast component of the band. |
| `slow_period` | 60 | Lookback for the slow component of the band. |
| `ema_period` | 20 | The signal line used to confirm the breakout direction. |
| `mabw_llv_period` | 10 | The "Squeeze" lookback; defines how long volatility must be low. |
| `mab_width_critical` | 30 | The "Blow-off" threshold; forces an exit when bands widen too far. |

---

## Market Hypothesis Validation

*   **Why it works:** Markets spend the majority of time in noise (mean reversion). By filtering for "Squeezes" (Width at LLV), the strategy avoids trading during choppy, high-volatility sideways markets.
*   **Risk Factors:**
    *   **Fakeouts:** In prolonged low-volatility regimes, price may breach the bands without sustaining a trend.
    *   **Lag:** Using moving averages inherently introduces lag; the breakout is identified after the move has begun.


Here is the revised documentation for the **MABW Strategy**, updated to reflect the specific logic in `strategy_mabw.py` and the empirical results provided in the backtest logs.

**Key Observations from Analysis:**
*   **High Performance / Low Frequency:** The strategy achieved a 283% return but only executed 5 trades in 5 years. This indicates a lack of statistical significance.
*   **Validation Failure:** The 100% Sharpe degradation and "Poor" parameter clustering indicate the strategy is currently **overfitted** to specific historical anomalies (likely the NVDA/AAPL trends) and is **not robust** for production without modification.


# MABW: Volatility Breakout Strategy
{: .fs-9 }

A trend-following system designed to capture explosive moves following periods of volatility compression ("Squeezes").
{: .fs-6 .fw-300 }

---

## Strategy Profile

| Metric | Value |
| :--- | :--- |
| **Logic Class** | Volatility Expansion |
| **Primary Tickers** | AAPL, NVDA, BIIB, JPM, QQQ |
| **Validation Status** | <span class="label label-red">Prototype (High Overfitting)</span> |
| **Trade Frequency** | Ultra-Low (Avg 1 trade/year) |

### Abstract
The MABW (Moving Average Band Width) strategy operates on the principle of **volatility clustering**. It assumes that extended periods of low volatility (compression) are precursors to significant trend expansions. The strategy waits for the bandwidth between a Fast and Slow MA to hit a historical low (LLV), then enters when momentum (EMA) confirms a breakout.

---

## Signal Logic

The logic is implemented via `strategy_mabw.py` using strictly vectorized boolean operations.

### 1. Indicator Calculation
*   **MABW Bands:** Defined by the spread between a Fast MA and a Slow MA, scaled by a multiplier.
*   **Width:** $$ \text{Width} = \text{UpperBand} - \text{LowerBand} $$
*   **Regime Filter (LLV):** The Lowest Low Value of the `Width` over $$ N $$ periods.
*   **Trigger (EMA):** An Exponential Moving Average of the Close price.

### 2. Entry Logic (Long Only)
A position is opened if and only if **both** conditions are true simultaneously:
1.  **Squeeze Condition:** The current Band Width is at its $$ N $$-period low (within floating point tolerance).
    $$ \text{Width}_t \le \text{LLV}(\text{Width}, \text{Period}_{LLV}) $$
2.  **Momentum Trigger:** The signal EMA crosses *above* the Upper Band.
    $$ \text{EMA}_t > \text{Band}_{Upper} \land \text{EMA}_{t-1} \le \text{Band}_{Upper, t-1} $$

### 3. Exit Logic
The trade is closed when volatility becomes excessive, suggesting trend exhaustion:
*   **Blow-off:** The Band Width crosses above a critical fixed threshold.
    $$ \text{Width}_t > \text{Critical}_{Width} $$

---

## Configuration & Performance

The following configuration was identified as the global optimum but shows signs of overfitting due to low sample size.

### Optimized Parameters
```yaml
parameters:
  fast_period: 45
  slow_period: 88
  multiplier: 0.72       # Band width scaling
  ema_period: 12         # Fast signal line
  mabw_llv_period: 27    # Squeeze lookback
  mab_width_critical: 50 # Exit threshold
  atr_period: 23
  atr_multiplier: 2.61
```

### Backtest Results (In-Sample)
*   **Total Return:** 283.52%
*   **Sharpe Ratio:** 0.96
*   **Max Drawdown:** -35.40%
*   **Win Rate:** 80.00% (4/5 trades)
*   **Profit Factor:** 43.73

> **Warning:** These metrics are derived from only **5 trades** over the historical period. The statistical significance is negligible.

---

## Robustness & Validation Analysis

The strategy failed the Walk-Forward Validation phase, indicating high sensitivity to specific market conditions.

### 1. Degradation Analysis
*   **Sharpe Degradation:** **100.00%**
*   *Interpretation:* The strategy failed to generate positive risk-adjusted returns in the Out-of-Sample (OOS) data. It likely "froze" (made no trades) or lost money consistently when applied to unseen data.

### 2. Parameter Stability (Cluster Analysis)
The optimization surface is highly unstable, with most parameters showing "Poor" clustering (CV > 0.30). This suggests the optimizer found a "needle in a haystack" rather than a robust parameter zone.

| Parameter | Stability Assessment | Importance (MDI) |
| :--- | :--- | :--- |
| `ema_period` | **Poor** (CV: 0.44) | **High (54%)** |
| `multiplier` | **Poor** (CV: 0.40) | **High (36%)** |
| `slow_period` | Moderate (CV: 0.21) | Low (0%) |

### 3. Critical Failures
1.  **Sample Size:** With only 5 trades, the Win Rate of 80% is statistically meaningless.
2.  **Sensitivity:** The strategy is 90% driven by `ema_period` and `multiplier`. The high CV on these parameters means slight changes (e.g., changing EMA from 12 to 13) could destroy performance.

### Recommendation
**REJECT for Production.** The strategy requires:
1.  Relaxing the `mabw_llv_period` to increase trade frequency.
2.  Adding a secondary regime filter (e.g., Volume or ADX) to improve robustness.
3.  Re-optimizing with a focus on maximizing `Trade Count` alongside Sharpe Ratio.


Here is the revised documentation for the **MABW Strategy**, updated to reflect the specific logic in `strategy_mabw.py` and the empirical results provided in the backtest logs.

**Key Observations from Analysis:**
*   **High Performance / Low Frequency:** The strategy achieved a 283% return but only executed 5 trades in 5 years. This indicates a lack of statistical significance.
*   **Validation Failure:** The 100% Sharpe degradation and "Poor" parameter clustering indicate the strategy is currently **overfitted** to specific historical anomalies (likely the NVDA/AAPL trends) and is **not robust** for production without modification.


# MABW: Volatility Breakout Strategy
{: .fs-9 }

A trend-following system designed to capture explosive moves following periods of volatility compression ("Squeezes").
{: .fs-6 .fw-300 }

---

## Strategy Profile

| Metric | Value |
| :--- | :--- |
| **Logic Class** | Volatility Expansion |
| **Primary Tickers** | AAPL, NVDA, BIIB, JPM, QQQ |
| **Validation Status** | <span class="label label-red">Prototype (High Overfitting)</span> |
| **Trade Frequency** | Ultra-Low (Avg 1 trade/year) |

### Abstract
The MABW (Moving Average Band Width) strategy operates on the principle of **volatility clustering**. It assumes that extended periods of low volatility (compression) are precursors to significant trend expansions. The strategy waits for the bandwidth between a Fast and Slow MA to hit a historical low (LLV), then enters when momentum (EMA) confirms a breakout.

---

## Signal Logic

The logic is implemented via `strategy_mabw.py` using strictly vectorized boolean operations.

### 1. Indicator Calculation
*   **MABW Bands:** Defined by the spread between a Fast MA and a Slow MA, scaled by a multiplier.
*   **Width:** $$ \text{Width} = \text{UpperBand} - \text{LowerBand} $$
*   **Regime Filter (LLV):** The Lowest Low Value of the `Width` over $$ N $$ periods.
*   **Trigger (EMA):** An Exponential Moving Average of the Close price.

### 2. Entry Logic (Long Only)
A position is opened if and only if **both** conditions are true simultaneously:
1.  **Squeeze Condition:** The current Band Width is at its $$ N $$-period low (within floating point tolerance).
    $$ \text{Width}_t \le \text{LLV}(\text{Width}, \text{Period}_{LLV}) $$
2.  **Momentum Trigger:** The signal EMA crosses *above* the Upper Band.
    $$ \text{EMA}_t > \text{Band}_{Upper} \land \text{EMA}_{t-1} \le \text{Band}_{Upper, t-1} $$

### 3. Exit Logic
The trade is closed when volatility becomes excessive, suggesting trend exhaustion:
*   **Blow-off:** The Band Width crosses above a critical fixed threshold.
    $$ \text{Width}_t > \text{Critical}_{Width} $$

---

## Configuration & Performance

The following configuration was identified as the global optimum but shows signs of overfitting due to low sample size.

### Optimized Parameters
```yaml
parameters:
  fast_period: 45
  slow_period: 88
  multiplier: 0.72       # Band width scaling
  ema_period: 12         # Fast signal line
  mabw_llv_period: 27    # Squeeze lookback
  mab_width_critical: 50 # Exit threshold
  atr_period: 23
  atr_multiplier: 2.61
```

### Backtest Results (In-Sample)
*   **Total Return:** 283.52%
*   **Sharpe Ratio:** 0.96
*   **Max Drawdown:** -35.40%
*   **Win Rate:** 80.00% (4/5 trades)
*   **Profit Factor:** 43.73

> **Warning:** These metrics are derived from only **5 trades** over the historical period. The statistical significance is negligible.

---

## Robustness & Validation Analysis

The strategy failed the Walk-Forward Validation phase, indicating high sensitivity to specific market conditions.

### 1. Degradation Analysis
*   **Sharpe Degradation:** **100.00%**
*   *Interpretation:* The strategy failed to generate positive risk-adjusted returns in the Out-of-Sample (OOS) data. It likely "froze" (made no trades) or lost money consistently when applied to unseen data.

### 2. Parameter Stability (Cluster Analysis)
The optimization surface is highly unstable, with most parameters showing "Poor" clustering (CV > 0.30). This suggests the optimizer found a "needle in a haystack" rather than a robust parameter zone.

| Parameter | Stability Assessment | Importance (MDI) |
| :--- | :--- | :--- |
| `ema_period` | **Poor** (CV: 0.44) | **High (54%)** |
| `multiplier` | **Poor** (CV: 0.40) | **High (36%)** |
| `slow_period` | Moderate (CV: 0.21) | Low (0%) |

### 3. Critical Failures
1.  **Sample Size:** With only 5 trades, the Win Rate of 80% is statistically meaningless.
2.  **Sensitivity:** The strategy is 90% driven by `ema_period` and `multiplier`. The high CV on these parameters means slight changes (e.g., changing EMA from 12 to 13) could destroy performance.

### Recommendation
**REJECT for Production.** The strategy requires:
1.  Relaxing the `mabw_llv_period` to increase trade frequency.
2.  Adding a secondary regime filter (e.g., Volume or ADX) to improve robustness.
3.  Re-optimizing with a focus on maximizing `Trade Count` alongside Sharpe Ratio.
