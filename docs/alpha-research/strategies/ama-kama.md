---
layout: default
title: AMA-KAMA Strategy
parent: Strategies
nav_order: 3
permalink: /docs/alpha-research/strategies/ama-kama/
---

# AMA-KAMA: Dual Adaptive Momentum
{: .fs-7 }

A trend-reversion system that pairs two variations of Perry Kaufman's Adaptive Moving Average to identify high-fidelity entries, filtered by volatility regimes.
{: .fs-5 .fw-300 }

[View Script](/notebooks/alpha-research/strategies/02-strategy-smi.html){:target="_blank" rel="noopener noreferrer"} -  [Github Repository](https://github.com/xxxxyyyy80008/systematic-trading-strategies){:target="_blank" rel="noopener noreferrer"}
{: .fs-2 .fw-300 }

---

## Strategy Profile

| Metric | Value |
| :--- | :--- |
| **Logic Class** | Dual Adaptive Trend / Mean Reversion |
| **Validation Status** | <span class="label label-green">Robust (35.15% Degradation)</span> |
| **Best Optimization Score** | 1.3810 (Sharpe) |
| **Global Holdout Sharpe** | **1.58** (2023–2025) |
| **Win Rate** | 74.19% |

### Abstract
This strategy moves beyond standard static trendlines by pitting two adaptive indicators against each other:
1.  **AMA (Adaptive Moving Average):** The primary "Fast" signal line, highly sensitive to efficiency ratios.
2.  **KAMA (Kaufman Adaptive Moving Average):** The "Slow" baseline, acting as the trend anchor.

By using two adaptive averages, the strategy dynamically adjusts its crossover sensitivity. In chop/noise, both lines flatten, reducing false signals. In trending environments, both align. The system enters on momentum crossovers (AMA > KAMA) strictly when the asset is in a "value" state (low RSI) and exits on **either** trend reversal or overextension.

---

## Signal Logic Specification

The strategy employs a **State + Trigger** mechanism for entry, and a flexible **Multi-Condition** mechanism for exit.

### 1. Indicators
*   **AMA:** Period 18 (Fast smoothing 3, Slow 23).
*   **KAMA:** Period 22 (Fast 3, Slow 34, ER Period 10).
*   **RSI:** Period 12.

### 2. Entry Logic (Long)
A long position is initiated when the faster adaptive momentum overtakes the adaptive baseline, confirmed by a recent oversold condition.
*   **Trigger:** The AMA crosses **above** the KAMA.
*   **State Filter:** The RSI is (or was within the last 3 bars) below the **Entry Threshold** (e.g., 45).

$$ \text{Entry} = (\text{AMA} \uparrow \text{KAMA}) \land (\text{Max}(\text{RSI}_{t-2} \dots \text{RSI}_t) < \text{Threshold}_{\text{Entry}}) $$

### 3. Exit Logic
The trade is closed when **either** the trend breaks **or** the asset becomes overextended. This logical `OR` condition helps secure profits during sharp spikes before a reversal signal occurs.
*   **Condition A (Trend Break):** The AMA crosses **below** the KAMA.
*   **Condition B (Overbought):** The RSI rises above the **Exit Threshold** (e.g., 58).

$$ \text{Exit} = (\text{AMA} \downarrow \text{KAMA}) \lor (\text{RSI}_t > \text{Threshold}_{\text{Exit}}) $$

---

## Optimized Configuration & Parameter Stability

The strategy was tuned via Walk-Forward Analysis. The results highlight a preference for a "Swing Trading" cadence, indicated by the tight RSI bands.

| Parameter | Optimized Value | Role | Stability (CV) | Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **ama_period** | **18** | Fast Signal Lookback | 0.035 | <span style="color:green">**Excellent**</span> |
| **kama_period** | **22** | Baseline Trend Lookback | 0.048 | <span style="color:green">**Excellent**</span> |
| **rsi_entry_max** | **45** | Undervalued Zone Limit | 0.011 | <span style="color:green">**Excellent**</span> |
| **rsi_exit_min** | **58** | Profit Take Threshold | 0.012 | <span style="color:green">**Excellent**</span> |

### Key Insights
*   **Dual Adaptive Stability:** Both `ama_period` and `kama_period` show excellent stability (CV < 0.05), proving that the edge relies on the *interaction* between these two specific adaptive calculations rather than curve-fitting one specific line.
*   **Aggressive Profit Taking:** The exit threshold of **58** is notably low. Coupled with the logical `OR` operator in the exit logic, this confirms the strategy is designed to capture the "meat" of the move and exit early, rather than riding a trend to its absolute peak.

---

## Robustness Analysis

### 1. Degradation Analysis
*   **Avg Sharpe Degradation:** <span style="color:orange">**35.15%**</span>
*   **Assessment:** **Robust.**
    *   The strategy retains roughly 65% of its in-sample performance on unseen data. Given the complexity of dual adaptive averages, this is a strong result.

### 2. Parameter Importance
*   **Dominant Factors:** While the EMA period showed high importance in initial scans, the shift to **KAMA** as the baseline makes the `rsi_entry_max` (**18.5%**) and the interaction between `ama_slow` and `kama_slow` smoothing constants the critical drivers of alpha.

---

## Global Holdout Performance (2023–2025)

The strategy delivered exceptional metrics in the holdout period, driven by the new logical `OR` exit which likely locked in gains during the volatility of 2023-2024.

| Metric | Result |
| :--- | :--- |
| **Total Return** | **51.65%** |
| **Sharpe Ratio** | **1.58** |
| **Sortino Ratio** | **2.38** |
| **Max Drawdown** | **-10.62%** |
| **Win Rate** | **74.19%** |
| **Profit Factor** | **25.73** |

### Trade Statistics
*   **Total Trades:** 31
*   **Avg Trade:** $5,011.69
*   **Best/Worst:** +$24,354 / -$1,474

### ⚠️ Performance Note
The **Profit Factor (25.73)** is outlier-high, suggesting the `OR` based exit logic perfectly avoided major drawdowns during this specific market regime. While the Win Rate (74%) is sustainable, expect the Profit Factor to normalize (lower) in choppy, sideways markets where the adaptive averages may whip-saw before the RSI exit is hit.
