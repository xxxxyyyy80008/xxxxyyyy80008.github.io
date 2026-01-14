# Parameter Stability Analysis

## The Fragility Problem

**A strategy is only as robust as its most sensitive parameter.**

Consider two strategies, both with 2.0 Sharpe ratios in backtesting:

**Strategy A:** Sharpe = 1.95-2.05 across lookback periods of 10-30 days  
**Strategy B:** Sharpe = 2.0 at exactly 18 days, drops to 0.5 at 17 or 19 days

Which would you deploy with real capital?

**Parameter stability analysis** distinguishes robust strategies from curve-fitted noise.

---

## Why This Matters

### The Optimization Paradox

Every optimization process faces a fundamental tradeoff:

```
High Optimization Intensity
        ↓
  Better In-Sample Fit
        ↓
  Higher Parameter Precision Required
        ↓
  Lower Out-of-Sample Robustness
```

**Stable parameters indicate** you've found a genuine market inefficiency that persists across regimes.

**Unstable parameters indicate** you've optimized to historical noise that won't persist.

---

## Quantitative Measures

### 1. Coefficient of Variation (CV)

**Definition:**
$$
CV = \frac{\sigma}{\mu} = \frac{\text{Standard Deviation}}{\text{Mean}}
$$

**Interpretation:**

| CV Range | Assessment | Implication |
|----------|------------|-------------|
| **< 0.10** | Highly Stable | Parameter is structural, regime-independent |
| **0.10 - 0.15** | Stable | Acceptable for production deployment |
| **0.15 - 0.25** | Moderate | Monitor closely, consider adaptive approach |
| **0.25 - 0.40** | Unstable | High overfitting risk, simplify strategy |
| **> 0.40** | Highly Unstable | Parameter is noise-fitted, abandon or restart |

**Why CV instead of Standard Deviation?**

CV is scale-independent. A parameter with μ=100, σ=10 has the same CV as μ=10, σ=1.

```python
def calculate_cv(values: List[float]) -> float:
    """
    Calculate coefficient of variation.
    
    Returns infinity if mean is zero (avoid division by zero).
    """
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std
    return std / mean if mean != 0 else np.inf
```

---

### 2. Range Ratio

**Definition:**
$$
\text{Range Ratio} = \frac{\max(\theta) - \min(\theta)}{\text{median}(\theta)}
$$

**Interpretation:**

| Range Ratio | Assessment | Example |
|-------------|------------|---------|
| **< 0.30** | Tight | Lookback: 18-22 days (median=20) |
| **0.30 - 0.60** | Moderate | Lookback: 15-25 days (median=20) |
| **0.60 - 1.00** | Wide | Lookback: 10-30 days (median=20) |
| **> 1.00** | Extreme | Lookback: 5-45 days (median=20) |

**Advantage:** More intuitive than CV for discrete parameters.

```python
def calculate_range_ratio(values: List[float]) -> float:
    """Calculate parameter range relative to median."""
    median = np.median(values)
    return (max(values) - min(values)) / median if median != 0 else np.inf
```

---

### 3. Temporal Autocorrelation

**Question:** Do optimal parameters show momentum or mean reversion?

```python
def calculate_param_autocorrelation(
    values: List[float], 
    lag: int = 1
) -> float:
    """
    Calculate autocorrelation of parameter series.
    
    Positive: Parameters show momentum (drift)
    Negative: Parameters mean-revert (oscillate)
    Near-zero: Parameters are random walk
    """
    return pd.Series(values).autocorr(lag=lag)
```

**Interpretation:**

| Autocorrelation | Pattern | Implication |
|-----------------|---------|-------------|
| **> +0.5** | Strong Trend | Parameter is adapting to regime shift |
| **0 to +0.5** | Weak Trend | Normal variation around mean |
| **-0.5 to 0** | Mean Reversion | Parameter oscillates (good for adaptive) |
| **< -0.5** | Strong MR | Possible overcompensation in optimization |

---

Here is the refined text for **Parameter Stability Analysis**, strictly aligned with the implementation details of **Method 2 (Global Optimization)**.

It focuses on the robustness of the singular global parameter set and the distribution of the top-performing trials, removing code, case studies, and irrelevant walk-forward metrics (like temporal autocorrelation).

***

# Parameter Stability Analysis

## The Robustness Objective

In the context of Global Optimization (Method 2), parameter stability is defined not by how parameters evolve over time, but by the properties of the solution space itself. A strategy is considered robust if the selected parameters reside within a "broad peak" of performance rather than a "narrow spike."

The objective of this analysis is to distinguish between parameters that capture genuine market inefficiencies (structural alpha) and those that are merely artifacts of curve-fitting to specific historical noise (spurious alpha).

---

## 1. Top Trials Distribution Analysis (Cluster Stability)

Method 2 identifies a single set of parameters intended to function across all historical windows. To assess the reliability of this selection, the system analyzes the spatial distribution of the top $$N$$ best-performing trials (typically the top 10 to 50 iterations from the Optuna study).

### The Logic of Clustering
We posit that in a robust solution space, the best-performing parameter sets should cluster closely together.

*   **Tight Clustering:** If the top 20 trials possess nearly identical parameter values, it indicates a stable convex optimization surface. The strategy is robust to minor execution errors or slippage.
*   **Scattered Distribution:** If the top 20 trials feature widely divergent parameter values (e.g., one trial uses a lookback of 20, another uses 80), it suggests the objective function is "noisy" or multimodal. The "best" result is likely a statistical outlier.

### Quantitative Metrics
To quantify this clustering, the system calculates dispersion metrics for the parameter values of these top trials:

*   **Coefficient of Variation (CV):**
    $$ CV = \frac{\sigma_{trials}}{\mu_{trials}} $$
    A low CV ($$< 0.10$$) among top trials confirms that the optimizer consistently converged on the same region. High CV suggests random luck.
*   **Range Ratio:**
    $$ R = \frac{\max(\theta) - \min(\theta)}{\text{median}(\theta)} $$
    This measures the width of the optimality plateau. A narrow range implies high sensitivity (fragility), while a moderate range implies a forgiving parameter space.

---

## 2. Cross-Window Performance Variance

Since the optimization process applies a single parameter set across three distinct historical windows (e.g., 2018-2019, 2019-2021, 2021-2022), stability is measured by the consistency of the objective score across these regimes.

### Metric: Standard Deviation of Windows
The system calculates the standard deviation of the objective scores ($$\sigma_{score}$$) achieved in each of the rolling windows for a given trial.

*   **Low Variance:** The strategy performs comparably in all three windows, indicating insensitivity to specific market regimes (e.g., Bull vs. Bear).
*   **High Variance:** The strategy generates exceptional returns in one window but poor returns in another. Even if the *average* score is high, such parameters are penalized or discarded to prevent regime-specific overfitting.

### Metric: Minimum Window Score
To enforce a "safety first" approach, the stability analysis prioritizes the **Minimum Window Score** ($$\min(S_{w1}, S_{w2}, S_{w3})$$) rather than the mean. A parameter set is only as robust as its worst historical performance.

---

## 3. Parameter Sensitivity (Importance Analysis)

Not all parameters contribute equally to strategy performance. Understanding sensitivity is crucial for dimensionality reduction and risk management.

### Mean Decrease Impurity (MDI)
The system utilizes the MDI method (via Random Forest regression on the trial history) to quantify the "importance" of each parameter.

*   **High Importance:** Parameters with high MDI values are the primary drivers of the strategy's variance. These require precise tuning and rigorous stability checks.
*   **Low Importance:** Parameters with near-zero MDI are "noise." In future iterations, these parameters can be fixed to constant values to reduce the dimensionality of the search space without degrading performance.

---

## 4. Degradation Analysis (In-Sample vs. Out-of-Sample)

Before final selection, the system performs a degradation check comparing the Training (Optimization) performance against the Testing (Out-of-Sample) performance within the historical windows.

$$
\text{Degradation} = \frac{\text{Metric}_{Test} - \text{Metric}_{Train}}{\text{Metric}_{Train}}
$$

*   **Acceptable Degradation:** A moderate drop in performance (e.g., -10% to -20%) is expected and statistically normal.
*   **Critical Failure:** A sharp decoupling (e.g., > -50% drop) or a sign inversion (profit becoming loss) flags the parameter set as overfit, regardless of its raw score. These sets are pruned from the candidate list.

---

## 5. Visual Confirmation Methods

Quantitative metrics are supplemented by visual inspection of the solution space to confirm the "Broad Peak" hypothesis.

### Parameter Heatmaps
2D contour plots visualize the interaction between two high-importance parameters (e.g., Lookback Period vs. Entry Threshold).
*   **Target:** A large, contiguous "hot zone" (green/yellow) indicating a wide region of profitability.
*   **Avoid:** "Islands" of profitability surrounded by poor performance, indicating a fragile fit.

### Drawdown Sensitivity Surfaces
These plots map how Maximum Drawdown changes as a function of parameter variation. A robust strategy should show a flat or gently sloping surface. Steep cliffs in the drawdown surface indicate that a small parameter shift (e.g., market behavior changing slightly) could lead to catastrophic risk.