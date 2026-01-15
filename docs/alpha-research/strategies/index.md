---
layout: default
title: Strategies
parent: Alpha Research & Signal Generation
nav_order: 3
has_children: true
has_toc: true
permalink: /docs/alpha-research/strategies/
---

# Systematic Trading Strategies & Backtesting
{: .fs-7 }

Systematic trading algorithms validated through Walk-Forward Analysis and Regime Stress Testing.
{: .fs-5 .fw-300 }



## Performance Matrix

The following table summarizes the current status of the strategy pipeline. Strategies are graded based on their Out-of-Sample (OOS) robustness and parameter stability.

| Strategy | Logic Class | Validation Status | Risk Profile |Script|
| :--- | :--- | :--- | :--- |:--- |
| **[SMI Momentum](./smi/)** | Mean Reversion / Oscillator | <span class="label label-green">Very Robust (-19.4% Degradation)</span>|  Moderate (High Win Rate) |[View Script](/notebooks/alpha-research/strategies/02-strategy-smi.html){:target="_blank" rel="noopener noreferrer"}|
| **[AMA-KAMA](./ama-kama/)** | Dual Adaptive Trend / Mean Reversion | <span class="label label-green">Robust (35.66% Degradation)</span> | Conservative  |[View Script](/notebooks/alpha-research/strategies/05-strategy-ama-kama.html){:target="_blank" rel="noopener noreferrer"}|
| **[MABW](./mabw/)** | Volatility Expansion / Breakout | <span class="label label-red">Rejected (100% Degradation)</span> | Aggressive / Unstable |[View Script](/notebooks/alpha-research/strategies/01-strategy-mabw.html){:target="_blank" rel="noopener noreferrer"}|

---

## Strategy Summaries

### 1. AMA-KAMA (Dual Adaptive Momentum)
The flagship robust strategy of this suite. It utilizes two variations of Perry Kaufman's Adaptive Moving Average (AMA) to create a dynamic crossover system.
*   **Concept:** Uses a "Fast" AMA to detect signal and a "Slow" KAMA as a baseline. The system adapts its speed based on market efficiency (noise), reducing trades during chop.
*   **Key Feature:** Integrates an RSI "Value" filter, entering trends only when the asset is locally oversold, effectively combining trend following with mean reversion.
*   **Status:** High robustness with low parameter degradation (35%).

### 2. SMI (Stochastic Momentum Index)
A classic momentum oscillator strategy used as a performance benchmark.
*   **Concept:** derived from the Stochastic Oscillator, the SMI creates a cleaner signal by using the distance of the close relative to the midpoint of the high/low range.
*   **Key Feature:** Identifies inflection points in momentum to capture the "meat" of a swing trade.
*   **Status:** Serves as the baseline for comparing complex adaptive logic.

### 3. MABW (Moving Average Band Width)
A volatility expansion system based on the Mandelbrot "clustering" hypothesis.
*   **Concept:** Monitors the spread between a Fast and Slow MA. When the spread compresses to historical lows (a "Squeeze"), it anticipates an explosive breakout.
*   **Key Feature:** Pure regime-based entry (Squeeze + Momentum Trigger).
*   **Status:** <span style="color:red">**Failed Validation.**</span> The strategy exhibited 100% performance degradation in out-of-sample testing, indicating severe overfitting to historical anomalies. It is documented here for research transparency but is not recommended for production.
```