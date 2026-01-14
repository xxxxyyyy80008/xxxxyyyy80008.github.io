---
layout: default
title: "Backtest Engine Architecture"
parent: Methodology
grand_parent: Alpha Research & Signal Generation
nav_order: 4
has_children: false
permalink: /docs/alpha-research/methodology/backteset-engine
---

# Backtest Engine Architecture

### **Design Philosophy: The Hybrid Approach**
Developing a backtesting engine requires balancing two competing constraints: **Computational Speed** (for optimization) and **Execution Granularity** (for realism).

My framework utilizes a **Hybrid Architecture**:
1.  **Vectorized Signal Generation:** Technical indicators and raw logic are computed across the entire dataframe in a single pass using Pandas/NumPy. This ensures $O(1)$ signal generation speed relative to time steps.
2.  **Event-Driven Execution:** The portfolio state, risk checks, and trade simulation occur sequentially in a time-stepped loop. This allows for realistic handling of path-dependent factors like trailing stops, capital depletion, and complex slippage models.

---

## 1. System Data Flow

The architecture decouples **Logic** (Strategy), **State** (Portfolio), and **Mechanics** (Execution). This separation of concerns ensures that a bug in the signal logic does not corrupt the position tracking math.

```mermaid
flowchart TD
    Data[Market Data (OHLCV)] -->|Vectorized| Strat[Abstract Strategy]
    Strat -->|Signals & Regimes| Eng[Backtest Engine Loop]
    
    subgraph "Event-Driven Loop (T -> T+1)"
    Eng -->|Daily State| PM[Portfolio Manager]
    PM -->|Allocation Logic| Risk[Risk & Constraints]
    Risk -->|Approved Orders| Exec[Trade Executor]
    Exec -->|Fills & Costs| PM
    end
    
    PM -->|Equity Curve| Results[Performance Metrics]
    
    style Strat fill:#f9f,stroke:#333,stroke-width:2px
    style PM fill:#bbf,stroke:#333,stroke-width:2px
    style Exec fill:#bfb,stroke:#333,stroke-width:2px
```

---

## 2. Core Components

### A. The Signal Generator (`strategy_base.py`)
All strategies inherit from an Abstract Base Class. This strictly defines the interface, requiring strategies to return standardized `Signal` objects.
*   **Vectorization:** Heavy lifting (e.g., Moving Averages, volatility bands) is pre-calculated.
*   **Safety:** The abstract class enforces the removal of `NaN` data before signal processing to prevent "pollution" of the signal stream.

### B. The Portfolio Manager (`portfolio_manager.py`)
This component acts as the "Central Bank" of the simulation. It holds the "Source of Truth" for the account balance.
*   **Dynamic Position Sizing:** Calculates size based on `TradeConfig` (e.g., % of Equity, Fixed Dollar).
*   **Rejection Logic:** Orders are rejected if they violate capital constraints or minimum trade size limits.
*   **Return Calculation:** Uses logarithmic returns for aggregation but reports arithmetic returns for final metrics.

### C. The Trade Executor (`trade_executor.py`)
Simulates the interaction with the market.
*   **Slippage Protocol:**
    *   Currently implements a **Fixed + Percentage** slippage model.
    *   $$ P_{exec} = P_{market} \cdot (1 \pm \delta_{pct}) \pm \delta_{fixed} $$
    *   *Correction:* In earlier iterations, slippage was double-counted on entry and exit calculation. The current version explicitly separates entry cost basis from exit realization.
*   **Commission:** Models per-share and per-order distinct costs.

---

## 3. Handling Edge Cases & Bias

A robust engine must handle data imperfections. My architecture explicitly addresses several common "silent killers" of backtests.

### Look-Ahead Bias Prevention
The engine strictly separates **Signal Time** from **Execution Time**.
*   **Signal:** Generated at Close of Day $T$.
*   **Execution:** Simulated at Open of Day $T+1$.
*   This prevents the strategy from "buying the low" of the day it generated the signal.

### Stale Order "Garbage Collection"
In a multi-asset universe, some assets may stop trading (delisting, halts) while the simulation continues.
*   **The Issue:** A limit order placed on a halted stock might "fill" weeks later at a stale price in a naive engine.
*   **The Fix:** The engine performs a `prune_stale_orders()` check at every time step. If a ticker is not present in the current day's data slice, any pending orders for that ticker are expired immediately.

### Safe Returns Math
To prevent runtime errors during the "Holdout" periods (where data might be sparse or volatile):
*   Implemented `np.errstate` context managers to handle division-by-zero scenarios gracefully.
*   Metrics like Calmar Ratio and Sortino Ratio default to `0.0` rather than `NaN` to allow the optimizer to continue seeking valid parameter sets.

---

## 4. Typed Data Structures

To ensure data integrity, I utilize Python `dataclasses` instead of loose dictionaries. This enforces strict typing across the stack.

```python
@dataclass(frozen=True)
class TradeConfig:
    """Immutable configuration to prevent runtime state modification"""
    initial_capital: float = 10000.0
    pos_size_type: str = 'percentage'  # 'fixed' or 'percentage'
    pos_size_value: float = 0.05       # e.g. 5% per trade
    commission_rate: float = 0.0005    # 5 bps
    slippage_pct: float = 0.0005       # 5 bps
    max_positions: int = 10
```

---

## 5. Future Roadmap

While the current engine is robust for daily/weekly strategies, I am working on the following extensions:

1.  **Market Impact Model:** Moving from linear slippage to a Square-Root Law model ($Cost \propto \sigma \sqrt{\frac{Q}{V}}$) to better simulate capacity constraints.
2.  **Intraday Support:** Adapting the time-step loop to handle minute-bar data for higher frequency strategies.