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
```