---
layout: default
title: SMI Strategy
parent: Strategies
nav_order: 2
permalink: /docs/alpha-research/strategies/smi/
---

# SMI: Stochastic Momentum Reversion
{: .fs-9 }

A rule-based mean-reversion strategy utilizing the **Stochastic Momentum Index (SMI)** to identify deep oversold conditions and execute entries upon momentum recovery.
{: .fs-6 .fw-300 }

---

## Strategy Profile

| Metric | Value |
| :--- | :--- |
| **Logic Class** | Mean Reversion / Oscillator |
| **Primary Tickers** | GS, MSFT, HD, V, SHW, CAT, MCD, UNH, AXP |
| **Validation Status** | <span class="label label-green">Very Robust (-19.4% Degradation)</span> |
| **Best Optimization Score** | 1.1200 (Sharpe) |
| **Global Holdout Sharpe** | 1.06 (2023–2025) |
| **Risk Profile** | Moderate (Counter-Trend Entry) |

### Abstract
The **Stochastic Momentum Index (SMI)** is a refinement of the classic Stochastic Oscillator. While the traditional Stochastic measures the close relative to the High/Low range (0 to 100), the SMI measures the distance of the close relative to the **midpoint** of the High/Low range (-100 to +100).

This strategy employs the SMI to detect "Deep Oversold" conditions—extreme deviations below the midpoint—and enters when momentum shifts back upward (signal line crossover). Recent optimization indicates that the strategy benefits from a **lower profit-taking threshold** (37) than typically expected, prioritizing higher win rates and faster turnover over extended holding periods.

---

## Signal Logic Specification

### 1. Indicator Calculation
1.  **Range:** Calculate Highest High and Lowest Low over $$k$$ periods.
2.  **Midpoint:** $$ \text{Mid} = (\text{High}_{max} + \text{Low}_{min}) / 2 $$.
3.  **Smoothing:** Calculate the difference ($$ \text{Close} - \text{Mid} $$) and smooth it twice using an EMA of period $$d$$.
4.  **SMI:** Normalize the smoothed difference against the smoothed range to yield a value between -100 and +100.

### 2. Entry Logic (Long)
A long entry is generated when the asset is deeply oversold and momentum turns positive.
*   **Filter:** SMI and Signal line must be below the **Oversold Threshold**.
*   **Trigger:** SMI crosses **above** the Signal line.

$$ \text{Entry} = (\text{SMI}_t > \text{Signal}_t) \land (\text{SMI}_{t-1} < \text{Signal}_{t-1}) \land (\text{SMI}_t < \text{Threshold}_{\text{OS}}) $$

### 3. Exit Logic
The trade is closed when upward momentum falters or reaches a target zone.
*   **Filter:** Signal line must be above the **Overbought Threshold**.
*   **Trigger:** SMI crosses **below** the Signal line.

$$ \text{Exit} = (\text{SMI}_t < \text{Signal}_t) \land (\text{Signal}_t > \text{Threshold}_{\text{OB}}) $$

---

## Optimized Configuration

The strategy was optimized using a Walk-Forward framework. The best-performing configuration (Score: 1.1200) emphasizes a longer lookback period and an asymmetric threshold structure.

| Parameter | Value | Role | Stability (CV) | Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **k_period** | **41** | Cycle Lookback | 0.090 | <span style="color:green">**Excellent**</span> |
| **d_period** | **2** | Signal Reactivity | 0.213 | <span style="color:red">**Poor**</span> |
| **oversold_threshold** | **-57** | Entry Filter | 0.081 | <span style="color:green">**Excellent**</span> |
| **overbought_threshold** | **37** | Exit Filter | 0.115 | <span style="color:orange">**Good**</span> |

### Key Insights
*   **Asymmetric Thresholds:** The optimal `overbought_threshold` (**37**) is significantly closer to zero than the `oversold_threshold` (**-57**). This creates a "Quick Exit" profile, where the strategy captures the initial mean reversion bounce rather than holding for a full trend reversal.
*   **Stable Core:** The high stability of `k_period` and `oversold_threshold` indicates that the strategy identifies a persistent market anomaly related to ~2-month cycles (41 days) and extreme deviations.
*   **Smoothing Sensitivity:** The `d_period` remains the most unstable parameter (CV 0.213), suggesting the strategy's exact entry timing is sensitive to signal noise.

---

## Validation & Robustness Analysis

### 1. Degradation Analysis
*   **Avg Sharpe Degradation:** <span style="color:green">**-19.42%**</span>
*   **Assessment:** **Very Robust.**
    *   A negative degradation value implies that the strategy performed **better** on average during Out-of-Sample (OOS) periods than during the In-Sample optimization phase. This is a strong indicator that the logic is not overfit to historical noise.

### 2. Parameter Importance (MDI)
Feature importance analysis reveals which parameters drive the strategy's alpha:

| Parameter | Importance | Interpretation |
| :--- | :--- | :--- |
| **k_period** | **44.13%** | The lookback window is the primary determinant of success. |
| **oversold_threshold** | **34.41%** | The entry filter level is the secondary driver. |
| **d_period** | 15.65% | Signal smoothing contributes moderately. |
| **overbought_threshold** | 5.81% | The specific exit level is the least important factor. |

---

## Global Holdout Performance (2023–2025)

The strategy was tested on a pristine holdout dataset (post-optimization) comprising the 9 primary tickers.

| Metric | Result |
| :--- | :--- |
| **Total Return** | **22.79%** |
| **Sharpe Ratio** | **1.06** |
| **Sortino Ratio** | **1.41** |
| **Max Drawdown** | **-7.62%** |
| **Win Rate** | **84.62%** |
| **Profit Factor** | **7.89** |

### Trade Statistics
*   **Total Trades:** 39
*   **Avg Trade:** $1,776.94
*   **Best/Worst:** +$7,929 / -$4,016


### Conclusion
The SMI strategy exhibits **Excellent** stability in its primary parameters and **Negative Degradation** in walk-forward testing, identifying it as a highly robust system. The high win rate (84%) in the holdout period is a direct result of the lower `overbought_threshold` (37), which successfully secures profits during initial mean-reversion impulses.
```