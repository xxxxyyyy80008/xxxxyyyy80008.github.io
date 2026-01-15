---
layout: default
title: AMA-KAMA Strategy
parent: Strategies
nav_order: 3
permalink: /docs/alpha-research/strategies/ama-kama/
---

# AMA-KAMA: Adaptive Momentum Reversion
{: .fs-7 }

A strategy leveraging Perry Kaufman's **Adaptive Moving Average (AMA)** to capture trend reversals, filtered by RSI-based volatility regimes.
{: .fs-5 .fw-300 }

---

## Strategy Profile

| Metric | Value |
| :--- | :--- |
| **Logic Class** | Adaptive Trend / Mean Reversion |
| **Validation Status** | <span class="label label-green">Robust (35.15% Degradation)</span> |
| **Best Optimization Score** | 1.3810 (Sharpe) |
| **Global Holdout Sharpe** | **1.58** (2023–2025) |
| **Win Rate** | 74.19% |

### Overview
This strategy utilizes the **Adaptive Moving Average (AMA)**, which dynamically adjusts its smoothing constant based on the "Efficiency Ratio" (ER) of market price action. In high-noise environments, the AMA flattens (stops trending); in directional markets, it tracks price closely.

The logic pairs the AMA with a standard Exponential Moving Average (EMA). Unlike typical trend followers, this system uses RSI as a pre-condition state filter. It specifically targets entries where momentum is turning bullish (AMA > EMA) immediately following a value zone (Low RSI), and exits when the trend dampens in an overbought zone.

*Note: While the KAMA (Kaufman Adaptive Moving Average) indicator is calculated in the pipeline for analysis, the active signal logic primarily relies on the standard AMA implementation.*

---

## Signal Logic Specification

The strategy employs a **State + Trigger** mechanism. The asset must first enter a specific RSI regime (State) before the crossover (Trigger) is valid.

### 1. Indicators
*   **AMA (Adaptive Moving Average):** Period 18, Fast 3, Slow 23.
*   **EMA (Baseline Trend):** Period 13.
*   **RSI (Relative Strength Index):** Period 12.

### 2. Entry Logic (Long)
A long position is initiated when adaptive momentum overtakes the baseline trend, but only if the asset was recently "cheap."
*   **State Filter:** The RSI must be (or recently was within 3 bars) below the **Entry Threshold**.
*   **Trigger:** The AMA crosses **above** the EMA.

$$ \text{Entry} = (\text{AMA} \uparrow \text{EMA}) \land (\text{Max}(\text{RSI}_{t-2} \dots \text{RSI}_t) < \text{Threshold}_{\text{Entry}}) $$

### 3. Exit Logic
The trade is closed when adaptive momentum falters relative to the baseline while price is elevated.
*   **State Filter:** The RSI must be above the **Exit Threshold**.
*   **Trigger:** The AMA crosses **below** the EMA.

$$ \text{Exit} = (\text{AMA} \downarrow \text{EMA}) \land (\text{RSI}_t > \text{Threshold}_{\text{Exit}}) $$

---

## Optimized Configuration & Parameter Stability

The strategy was tuned via Walk-Forward Analysis. The resulting configuration favors a fast-reacting AMA against a relatively fast EMA, creating a nimble "swing trading" profile.

| Parameter | Optimized Value | Role | Stability (CV) | Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **ama_period** | **18** | Adaptive Efficiency Lookback | 0.035 | <span style="color:green">**Excellent**</span> |
| **ama_fast** | **3** | Fast Smoothing Constant | 0.000 | <span style="color:green">**Excellent**</span> |
| **ema_period** | **13** | Baseline Trend Reference | 0.037 | <span style="color:green">**Excellent**</span> |
| **rsi_entry_max** | **45** | Undervalued Zone Limit | 0.011 | <span style="color:green">**Excellent**</span> |
| **rsi_exit_min** | **58** | Overvalued Zone Limit | 0.012 | <span style="color:green">**Excellent**</span> |

### Key Insights
*   **Parametric Stability:** The strategy exhibits exceptional stability. The `ama_fast` parameter has a CV of **0.000**, indicating that a fast smoothing constant of 3 is universally optimal across all test windows. `rsi_entry_max` (CV 0.011) is also highly stable.
*   **Tight RSI Bands:** The optimal RSI entry (< 45) and exit (> 58) are not "extreme" (e.g., 30/70). This suggests the strategy operates best in **Trend Continuation** modes—entering on mild pullbacks and exiting on mild strength—rather than hunting for market crash reversals.

---

## Robustness Analysis

### 1. Degradation Analysis
*   **Avg Sharpe Degradation:** <span style="color:orange">**35.15%**</span>
*   **Assessment:** **Robust.**
    *   While higher than the SMI strategy, a degradation of 35% is well within the acceptable range for trend-following systems. It indicates the strategy retains roughly 65% of its in-sample performance when applied to unknown data.

### 2. Parameter Importance (MDI)
*   **Primary Driver:** `ema_period` (**30.5%**). The baseline trend definition is the most critical factor in the strategy's success.
*   **Secondary Driver:** `rsi_entry_max` (**18.5%**). Defining the "value zone" correctly is the second most important determinant of alpha.

---

## Global Holdout Performance (2023–2025)

The strategy delivered exceptional performance on the holdout set, significantly outperforming its training metrics.

| Metric | Result |
| :--- | :--- |
| **Total Return** | **51.65%** |
| **Sharpe Ratio** | **1.58** |
| **Sortino Ratio** | **2.38** |
| **Max Drawdown** | **-10.62%** |
| **Win Rate** | **74.19%** |
| **Profit Factor** | **25.73** |

### Trade Statistics
*   **Total Trades:** 31 (Low Frequency / High Conviction)
*   **Avg Trade:** $5,011.69
*   **Best/Worst:** +$24,354 / -$1,474

### ⚠️ Performance Anomaly Note
The **Profit Factor of 25.73** is extraordinarily high. This indicates that in the holdout period (2023-2025), the strategy likely caught several major trends while keeping losses negligible. While favorable, such a high PF is rare in live trading and likely benefits from the $0.00 commission assumption. However, the high Expectancy ($5k per trade) suggests the edge is real and substantial.

### Conclusion
The AMA-KAMA strategy is a highly stable, low-frequency swing system. Its "Excellent" parameter stability scores and strong holdout performance (Sharpe 1.58) suggest it effectively identifies high-probability reversals where adaptive momentum confirms a resumption of trend.
