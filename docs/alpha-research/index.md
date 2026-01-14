---
layout: default
title: Alpha Research & Signal Generation
nav_order: 5
has_children: true
has_toc: false
permalink: /docs/alpha-research/
---

# Systematic Alpha Research Framework
{: .fs-7 }

A rigorous pipeline for the engineering, calibration, and stress-testing of systematic trading strategies.
{: .fs-5 .fw-300 }


[Github Repository](https://github.com/xxxxyyyy80008/systematic-trading-strategies){: .btn .btn-primary .fs-3 .mb-2 .mb-md-0}{:target="_blank" rel="noopener noreferrer"}
---


---

## Overview

This section documents the quantitative methodologies used to develop tradeable systematic strategies: constructing robust execution systems that can capture theoretical edges while withstanding market friction and regime shifts.

The research framework prioritizes **generalization** over raw performance. It employs a **Global Optimization** protocol designed to identify a single, stable parameter configuration that functions effectively across diverse historical periods, rather than overfitting parameters to specific market cycles.

---

## Infrastructure & Methodology

The reliability of these strategies relies on the integrity of the underlying validation engine.

### 1. [Walk-Forward Methodology](./framework/walkforward-analysis)
A detailed breakdown of the validation protocol.
*   **Global Robustness:** Searching for parameter sets that survive diverse market conditions.
*   **Degradation Analysis:** A quantitative check to reject strategies where $$ \text{Sharpe}_{IS} \gg \text{Sharpe}_{OOS} $$.

### 2. [Parameter Stability Analysis](./framework/parameter-stability)
Ensuring the strategy is not poised on a "knife-edge" of optimization.
*   **Sensitivity Mapping:** Verifying that small changes in inputs do not cause catastrophic failures in outputs.
*   **Convexity Check:** Prioritizing broad, flat regions of the solution space over narrow, high peaks.

### 3. [Backtest Engine Architecture](./framework/backtest-engine)
Technical documentation of the Python-based simulation engine.
*   **Hybrid Architecture:** Vectorized signal calculation for throughput ($$ O(1) $$) + Event-driven execution for path-dependent accuracy.
*   **Friction Modeling:** Implementation of slippage, commission, and stale-data pruning.

---

## Strategy Research Pipeline

The development lifecycle adheres to an institutional-grade workflow, strictly separating the **Calibration Phase** (Signal Design & Optimization) from the **Validation Phase** (Walk-Forward & Stress Testing).

```mermaid
graph TD
    subgraph "I. Ideation & Design"
        A[Economic Hypothesis] -->|Quantify| B[Signal Specification]
        B -->|Vectorization| C[Prototype Implementation]
    end

    subgraph "II. Calibration (In-Sample)"
        C --> D{Global Parameter Search}
        D -->|Optuna TPE| E[Cross-Window Optimization]
        E -->|Select| F[Candidate Parameter Set]
        F -->|Constraint| G[Maximize Regime Stability]
    end

    subgraph "III. Validation (Out-of-Sample)"
        G --> H[Walk-Forward Analysis]
        H -->|Check| I{Degradation Assessment}
        I -->|High Decay| J[Reject: Overfitted]
        I -->|Stable| K[Parameter Sensitivity Check]
        K --> L[Final Holdout Test]
    end

    style D fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#bfb,stroke:#333,stroke-width:2px
```

### Phase Details

1.  **Hypothesis & Design:** Strategies begin with a structural market premise (e.g., volatility clustering, mean reversion constraints). Signals are implemented using pure functions to ensure stateless reproducibility.
2.  **Global Parameter Search:** Instead of optimizing each window individually (which leads to curve-fitting), we search for a **single parameter set** that maximizes the robust objective function across *all* training windows simultaneously.
3.  **Walk-Forward Validation:** The candidate parameters are applied to unseen data. We measure **Performance Degradation**—the gap between In-Sample training results and Out-of-Sample test results—to quantify the strategy's "optimism bias."

---

## Strategy Universe

Strategies are classified by their underlying logic class and validated based on their Out-of-Sample (OOS) stability.

### Volatility & Regime Detection

| Strategy ID | Logic Class | Mathematical Premise | Validation Status |
| :--- | :--- | :--- | :--- |
| **[MABW](./strategies/mabw)** | **Volatility_Expansion** | Periods of low variance ($$\sigma^2$$) statistically precede high-variance expansion. Trends are captured via expansion thresholds. | **Production** <br> *(Low Degradation)* |
| **[VPN](./strategies/vpn)** | **Volume_Divergence** | Price displacement normalized by volume ($$P \times V$$) and volatility ($$ATR$$) isolates institutional flow from retail noise. | **Validation** <br> *(Stable OOS Sharpe)* |

### Adaptive Momentum

| Strategy ID | Logic Class | Mathematical Premise | Validation Status |
| :--- | :--- | :--- | :--- |
| **[AMA](./strategies/ama)** | **Fractal_Efficiency** | Trend persistence is a function of path efficiency. Lag parameters ($$\alpha$$) are dynamically adjusted based on the Efficiency Ratio ($$ER$$). | **Production** <br> *(High Regime Adaptation)* |
| **[RS-EMA](./strategies/rsema)** | **Relative_Strength** | Assets exhibiting relative strength ($$\beta > 1$$) vs. a benchmark demonstrate momentum persistence during trend continuations. | **Review** <br> *(Drawdown Sensitivity)* |

### Mean Reversion

| Strategy ID | Logic Class | Mathematical Premise | Validation Status |
| :--- | :--- | :--- | :--- |
| **[Stoch-MACD](./strategies/stmacd)** | **Oscillator_Confluence** | Reversal probability is maximized when unbounded momentum (MACD) diverges from bounded oscillation (Stochastic) at statistical extremes. | **Validation** <br> *(High Win Rate)* |

---



### Tech Stack

*   **Language:** Python 3.10+
*   **Core Libraries:** Pandas, NumPy (Vectorization)
*   **Optimization:** Optuna (Tree-structured Parzen Estimator)
*   **Validation:** Custom Walk-Forward Engine
*   **Visualization:** Matplotlib, Seaborn

