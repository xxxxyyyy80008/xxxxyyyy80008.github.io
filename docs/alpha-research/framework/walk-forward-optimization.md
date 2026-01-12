
# Walk-Forward Optimization Protocol

## The Challenge: Overfitting in Time-Series

In quantitative finance, standard cross-validation techniques (like k-fold) fail because they ignore the temporal nature of market data. Randomly shuffling data destroys serial correlation and introduces look-ahead bias, rendering backtests useless for predicting live performance.

**The "In-Sample Illusion":** A strategy optimized on the full dataset will almost always look profitable. This is curve-fitting, not research.

## My Solution: Anchored Walk-Forward Analysis

To emulate the life-cycle of a trading strategy, I implemented a strict **Walk-Forward Optimization (WFO)** engine. This approach repeatedly optimizes the strategy on a past window and tests it on a future "unseen" window, rolling forward through time.

### Window Architecture

I utilize a **rolling window** configuration to allow the strategy to adapt to changing market regimes while maintaining strict separation of data.

**Configuration:**
- **Training Window (In-Sample):** 12 months
- **Testing Window (Out-of-Sample):** 3 months
- **Step Size:** 3 months
- **Holdout Period:** Final 6 months (Completely locked box)

### Visualization of the Rolling Process

```text
Time ----------------------------------------------------------------------->

[  Training (12m)  ][ Test (3m) ]
           ↓
   Roll Forward (3m)
           ↓
             [  Training (12m)  ][ Test (3m) ]
                        ↓
                Roll Forward (3m)
                        ↓
                          [  Training (12m)  ][ Test (3m) ]
```

---

## Implementation Details

The WFO engine is built using **pure functional programming principles** to ensure state isolation. Each window is processed independently, guaranteeing that no information leaks from the future into the past.

### 1. Window Generation
The windows are generated immutably before any processing begins.

```python
@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

def generate_windows(dates: List[pd.Timestamp], ... ) -> List[WalkForwardWindow]:
    """
    Pure function generating valid train/test splits.
    Ensures strict temporal ordering.
    """
    # ... implementation handles date math ...
```

### 2. Bayesian Parameter Optimization (Optuna)
Inside each *Training* window, I use **Optuna** to find optimal parameters. Unlike grid search, Bayesian optimization efficiently explores high-dimensional parameter spaces.

```python
def optimize_parameters(train_data, config) -> Dict[str, Any]:
    """
    Uses Tree-structured Parzen Estimator (TPE) to find 
    optimal parameters maximizing the Sharpe Ratio.
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    return study.best_params
```

### 3. Out-of-Sample execution
The "winning" parameters from the training set are applied *blindly* to the test set. These test periods are then stitched together to form the **Walk-Forward Equity Curve**.

---

## Performance Evaluation: The Degradation Coefficient

The critical metric is not the absolute return, but the **Performance Degradation** between In-Sample (IS) and Out-of-Sample (OOS) results.

$$ D = 1 - \frac{\text{Metric}_{OOS}}{\text{Metric}_{IS}} $$

A robust strategy should have similar performance in both phases. High degradation indicates overfitting.

### Acceptance Criteria

| Degradation ($D$) | Interpretation | Action |
| :--- | :--- | :--- |
| **< 10%** | **Suspiciously Robust** | Investigate for look-ahead bias or data leakage. |
| **10% - 20%** | **Production Ready** | Strategy generalizes well. Standard friction. |
| **20% - 40%** | **Fragile** | Significant curve-fitting. Simplify model complexity. |
| **> 40%** | **Broken** | Pure noise mining. Reject strategy. |

---

## Real-World Results

Below is the aggregate analysis from the **MABW (Moving Average Bollinger Width)** strategy across 4 years of data (2020-2024).

### Statistical Summary

| Metric | Training Mean (IS) | Testing Mean (OOS) | Degradation |
| :--- | :--- | :--- | :--- |
| **Sharpe Ratio** | 2.12 | 1.83 | **13.7%** |
| **Sortino Ratio** | 3.05 | 2.55 | **16.4%** |
| **Win Rate** | 64.2% | 59.1% | **7.9%** |

### Parameter Stability Analysis
I also track how parameters evolve over time. High variance in optimal parameters suggests the strategy is chasing noise rather than a persistent anomaly.

*Example: Evolution of Lookback Period*
![Parameter Stability](../images/param_stability_mabw.png)

> **Insight:** The optimal 'Bollinger Period' remained stable between 18-22 days for 85% of windows, indicating a genuine structural market property rather than a fleeting regime.

---

## Functional Architecture Advantage

Why build this functionally?

1.  **Parallelization:** Because `process_window(window, data)` is a pure function with no side effects, I can run all rolling windows in parallel across CPU cores without race conditions.
2.  **Reproducibility:** A specific seed and configuration always yield the exact same parameters and trades, critical for institutional compliance.
3.  **Safety:** Immutable data structures prevent the "accidental look-ahead" bugs common in stateful OOP backtesters (e.g., accessing `df.iloc[i+1]` by mistake).

---

## Conclusion

This protocol filters out roughly 90% of strategy ideas that look good on a static backtest but fail to adapt. The surviving strategies demonstrated here have passed this rigorous gauntlet, providing a high degree of confidence in their potential for live trading.
```