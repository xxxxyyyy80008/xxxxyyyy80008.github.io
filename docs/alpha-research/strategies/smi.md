---
layout: default
title: SMI Strategy
parent: Strategies
nav_order: 2
permalink: /docs/alpha-research/strategies/smi/
---

# SMI: Stochastic Momentum Reversion
{: .fs-7 }

A highly robust mean-reversion system that identifies deep value entries within medium-term volatility cycles.
{: .fs-5 .fw-300 }

---

## Strategy Profile

| Metric | Value |
| :--- | :--- |
| **Logic Class** | Mean Reversion / Oscillator |
| **Primary Tickers** | GS, MSFT, HD, V, SHW, CAT, MCD, UNH, AXP (Top 9 account for ~50% of DJI weighting)|
| **Validation Status** | <span class="label label-green">Robust (Production Ready)</span> |
| **Performance Score** | **1.11 Sharpe** (Global Best) |
| **Risk Profile** | Low (Negative Degradation in OOS) |

### Abstract
The **Stochastic Momentum Index (SMI)** refines the traditional Stochastic Oscillator by measuring the close relative to the *midpoint* of the High/Low range rather than the absolute Low.

Unlike typical short-term oscillators, recent optimization indicates this strategy is most effective when tuned to **quarterly cycles** (41 days). It ignores minor noise and enters only when price is deeply overextended (Oversold < -58) and momentum immediately shifts positive.

---

## Signal Logic Specification

The strategy employs a precise sequence of market state detection followed by a momentum trigger.

### 1. Indicator Calculation
1.  **Range Calculation:** Determine the Highest High and Lowest Low over a lookback period $$k$$.
2.  **Midpoint Deviation:** Calculate the difference between the current Close and the Midpoint of that range.
3.  **Double Smoothing:** Apply an Exponential Moving Average (EMA) of period $$d$$ to the result, and then apply the EMA again. This eliminates noise lag.
4.  **Normalization:** The result is scaled between -100 and +100.

### 2. Entry Logic (Long)
A buy signal requires the convergence of extreme valuation and immediate momentum recovery:
*   **Deep Value Filter:** The SMI value must drop below a strict **Oversold Threshold** (e.g., -58). This ensures the asset is trading at a significant discount relative to its recent range.
*   **Momentum Crossover:** The SMI line must cross *above* its own Signal Line (EMA). This confirms the bottom has likely formed.

$$ \text{Entry} = (\text{SMI}_t > \text{Signal}_t) \land (\text{SMI}_{t-1} < \text{Signal}_{t-1}) \land (\text{SMI}_t < \text{Threshold}_{Oversold}) $$

### 3. Exit Logic
The trade is closed on trend exhaustion:
*   **Overextension:** The Signal line must be above the **Overbought Threshold** (e.g., 53).
*   **Momentum Loss:** The SMI line crosses *below* the Signal Line.

---

## Performance & Robustness Analysis

The strategy underwent rigorous Walk-Forward Analysis (WFA) on a basket of 9 blue-chip tickers. The results demonstrate exceptional robustness.

### Optimized Configuration

| Parameter | Value | Role | Stability |
| :--- | :--- | :--- | :--- |
| **k_period** | **41** | Cycle Lookback | **Excellent** |
| **d_period** | **2** | Smoothing Speed | Poor |
| **oversold_threshold** | **-58** | Entry Zone | **Excellent** |
| **overbought_threshold** | **53** | Exit Zone | Moderate |

### 1. Robustness (Degradation Analysis)
*   **Avg Sharpe Degradation:** <span style="color:green">**-11.01%**</span>
*   **Interpretation:** A negative degradation score indicates that the strategy performed *better* in Out-of-Sample (OOS) testing than in In-Sample optimization. This is a rare and strong signal that the logic captures a fundamental market behavior rather than noise.

### 2. Parameter Sensitivity & Importance
The optimization landscape reveals exactly what drives the strategy's edge:

*   **Primary Driver ($$ k $$ Period):** With **47% Importance** and **Excellent Stability** (CV 0.039), the Lookback Period is the critical "tuning knob." The optimal value of **41** suggests the strategy trades on quarterly earnings cycles or medium-term institutional flows, rather than weekly noise.
*   **The "Safety Valve" (Oversold Threshold):** The specific level of -58 is highly stable (**26% Importance**). This suggests a structural floor where mean reversion reliably occurs for these tickers.
*   **Exit Irrelevance:** The `overbought_threshold` has only **4% importance**. This implies the *entry* is the source of alpha; once in the trade, the specific exit level matters less than the momentum shift itself.

### Recommendation
**Status: APPROVED.**

The strategy shows signs of being a "Cycle Hunter." It waits patiently for a 2-month low (k=41) combined with a statistical anomaly (oversold < -58) before striking.
*   **Deployment:** Suitable for the analyzed basket (GS, MSFT, HD, etc.).
*   **Note:** The low `d_period` (2) indicates the need for fast reaction times once the setup is detected.


## Backtest on Global Holdout (2023-2025)
Tickers: ["GS", "MSFT", "HD", "V", "SHW", "CAT", "MCD", "UNH", "AXP"]
Top 9 - by index weighting - of DJI that account for 50% of weighting. 

Period: 2023-2025.

================================================================================
PERFORMANCE SUMMARY
================================================================================

*** RETURNS ***
  Initial Capital:    $  300,000.00
  Final Value:        $  321,271.25
  Total P&L:          $   21,271.25
  Total Return:               7.09%

*** RISK METRICS ***
  Sharpe Ratio:               1.04
  Sortino Ratio:              1.12
  Max Drawdown:              -2.63%
  Calmar Ratio:               2.70
  Annual Volatility:          2.21%

*** TRADE STATISTICS ***
  Total Trades:                 55
  Win Rate:                  72.73%
  Profit Factor:              3.74
  Expectancy:         $      417.44
  Best Trade:         $     4266.15
  Worst Trade:        $    -2823.92
  Average Trade:      $      417.44

*** COSTS ***
  Total Commission:   $        0.00
  Total Slippage:     $        0.00
  Total Costs:        $        0.00


Here is the refined documentation for the **SMI Strategy**, incorporating the Walk-Forward Analysis and Parameter Stability results.



# SMI: Stochastic Momentum Reversion
{: .fs-7 }

A mean-reversion strategy utilizing the Stochastic Momentum Index to identify and trade reversals from deep oversold territory.
{: .fs-5 .fw-300 }

---

## Strategy Profile

| Metric | Value |
| :--- | :--- |
| **Logic Class** | Mean Reversion / Oscillator |
| **Indicator** | Stochastic Momentum Index (SMI) |
| **Primary Tickers** | GS, MSFT, HD, V, SHW, CAT, MCD, UNH, AXP |
| **Validation Status** | <span class="label label-green">Validated (Robust OOS)</span> |
| **Risk Profile** | Moderate (Counter-Trend Entry) |
| **Edge Source** | Overextended Price Action |

### Abstract
The **Stochastic Momentum Index (SMI)** is a refinement of the classic Stochastic Oscillator. While the traditional Stochastic measures where the close is relative to the High/Low range (0 to 100), the SMI measures the distance of the close relative to the **midpoint** of the High/Low range (-100 to +100).

This strategy employs the SMI to detect "Deep Oversold" conditions—extreme deviations below the midpoint—and enters when momentum shifts back upward (signal line crossover). The strategy has been validated through Walk-Forward Analysis and demonstrates **exceptional robustness**, with Out-of-Sample performance exceeding In-Sample expectations.

---

## Mathematical Specification

### 1. Indicator Calculation
The SMI calculates the position of the close relative to the midpoint of the range, double-smoothed by EMAs to reduce noise.

1.  **High-Low Range ($$k$$ periods):**
    $$ \text{HighestHigh} = \text{MAX}(\text{High}, k) $$
    $$ \text{LowestLow} = \text{MIN}(\text{Low}, k) $$
2.  **Relative Difference:**
    $$ \text{Midpoint} = \frac{\text{HighestHigh} + \text{LowestLow}}{2} $$
    $$ \text{Diff} = \text{Close} - \text{Midpoint} $$
3.  **Double Smoothing ($$d$$ periods):**
    The strategy applies an EMA of period $$d$$, then applies it again to the result.
    $$ \text{Smooth}_{\text{Diff}} = \text{EMA}(\text{EMA}(\text{Diff}, d), d) $$
    $$ \text{Smooth}_{\text{Range}} = \text{EMA}(\text{EMA}(\text{HighestHigh} - \text{LowestLow}, d), d) $$
4.  **SMI Value:**
    $$ \text{SMI} = \frac{\text{Smooth}_{\text{Diff}}}{0.5 \times \text{Smooth}_{\text{Range}}} \times 100 $$
5.  **Signal Line:**
    $$ \text{Signal} = \text{EMA}(\text{SMI}, d) $$

---

## Signal Logic

The strategy uses specific vectorized boolean logic to generate signals.

### Entry Logic (Long)
A buy signal is generated when momentum turns positive while price is significantly oversold.
*   **Condition 1 (Crossover):** The SMI line crosses **above** the Signal line.
*   **Condition 2 (Deep Filter):** Both the SMI and the Signal line must be **below** the `oversold_threshold`.

$$ \text{Entry} = (\text{SMI}_t > \text{Signal}_t) \land (\text{SMI}_{t-1} < \text{Signal}_{t-1}) \land (\text{SMI}_t < \text{Threshold}_{\text{OS}}) $$

### Exit Logic
The trade is closed when momentum exhausts itself in overbought territory.
*   **Condition 1 (Crossunder):** The SMI line crosses **below** the Signal line.
*   **Condition 2 (Extension Filter):** The Signal line must be **above** the `overbought_threshold`.

$$ \text{Exit} = (\text{SMI}_t < \text{Signal}_t) \land (\text{Signal}_t > \text{Threshold}_{\text{OB}}) $$

---

## Optimized Configuration

The following parameters were identified through rigorous optimization and validated through Walk-Forward Analysis across 9 large-cap equities.

### Parameters

| Parameter | Value | Range Tested | Stability |
| :--- | :--- | :--- | :--- |
| **k_period** | 41 | [20, 60] | **Excellent** (CV: 0.039) |
| **d_period** | 2 | [2, 5] | Poor (CV: 0.213) |
| **oversold_threshold** | -58 | [-70, -40] | **Excellent** (CV: 0.048) |
| **overbought_threshold** | 53 | [40, 60] | Moderate (CV: 0.166) |

### Key Insights
*   **Longer Lookback:** The optimal `k_period` of 41 days is significantly longer than the traditional default (8-14 days). This suggests the strategy performs better when identifying structural oversold conditions rather than short-term noise.
*   **Deeper Threshold:** The `oversold_threshold` of -58 is more extreme than standard (-40 or -50). This acts as a **quality filter**, ensuring entries only occur during extreme dislocation.
*   **Fast Signal:** The `d_period` of 2 creates a highly reactive signal line, though this parameter shows poor clustering (high sensitivity).

---

## Validation & Robustness Analysis

The strategy was subjected to Walk-Forward Analysis with rolling optimization windows. The results demonstrate exceptional robustness.

### 1. Out-of-Sample Performance
*   **Best In-Sample Sharpe:** 1.1127
*   **Sharpe Degradation:** **-11.01%**
    *   *Interpretation:* The negative value indicates the strategy performed **11% BETTER** in Out-of-Sample data than during optimization. This is a rare and highly positive result, suggesting the logic captures a genuine market inefficiency rather than curve-fitting.

### 2. Parameter Stability (Clustering Analysis)

The optimization landscape was analyzed to determine if the "optimal" parameters represent a robust zone or a random spike.

| Parameter | Importance (MDI) | Stability | Assessment |
| :--- | :--- | :--- | :--- |
| **k_period** | **47.66%** | **Excellent** | The most critical parameter is highly stable across subsamples. |
| **oversold_threshold** | **26.39%** | **Excellent** | The second most important parameter is also robust. |
| **d_period** | 21.39% | Poor | Shows high variance, but contributes only ~20% of performance. |
| **overbought_threshold** | 4.57% | Moderate | Minimal impact on strategy performance. |

### 3. Critical Success Factors
The strategy's robustness is driven by two key structural elements:
1.  **Stable Core Logic:** The two most important parameters (`k_period` and `oversold_threshold`) both exhibit excellent clustering (CV < 0.05), indicating the strategy's edge is not dependent on precise parameter tuning.
2.  **Minimal Exit Sensitivity:** The exit threshold (`overbought_threshold`) accounts for less than 5% of performance variance, meaning the strategy is not "exit-dependent" and can tolerate a wide range of profit-taking rules.

### Recommendation
**Status: APPROVED for Production.**

The strategy demonstrates structural robustness and is suitable for live deployment with the following caveats:
*   **Monitor d_period Sensitivity:** While the smoothing period is less critical than the lookback, its poor stability suggests it may require periodic recalibration (quarterly review recommended).
*   **Diversification:** The strategy was tested on large-cap equities. Performance on small-cap or international markets should be validated before expansion.
