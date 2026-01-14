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
```