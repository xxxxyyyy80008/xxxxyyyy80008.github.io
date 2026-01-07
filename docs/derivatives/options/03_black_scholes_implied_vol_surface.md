# Black-Scholes Option Pricing - Implied Volatility Surface Analysis

**Notebooks**
- **[Github Notebook](https://github.com/xxxxyyyy80008/Option-Pricing-and-Risk-Models/blob/main/notebooks/03_black_scholes_implied_vol_surface.ipynb)**
- **[Kaggle Notebook](https://www.kaggle.com/code/xxxxyyyy80008/option-pricing-implied-volatility-surface/)**

##  Overview

This Jupyter notebook implements a comprehensive **Black-Scholes-Merton option pricing framework** with advanced **implied volatility (IV) surface analysis**, market data integration, and accuracy validation. The notebook extracts real-time options data, computes implied volatilities using robust numerical methods, and visualizes volatility surfaces in 2D and 3D.

---

##  Objectives

1. **Calculate theoretical option prices** using Black-Scholes-Merton with dividend yield adjustments
2. **Compute implied volatilities** from market prices using three numerical methods
3. **Integrate real market data** from Yahoo Finance API for validation
4. **Generate volatility surfaces** with 2D/3D visualizations and interpolation
5. **Validate pricing accuracy** through comprehensive error metrics

---

##  Key Features

### **Core Functionality**

| Feature | Description |
|---------|-------------|
| **Black-Scholes Pricing** | Call/put pricing with continuous dividend yield support |
| **Implied Volatility Calculation** | Three methods: Newton-Raphson, Brent's, and Hybrid |
| **Market Data Integration** | Live options chains from Yahoo Finance (yfinance) |
| **Volatility Surface Generation** | 2D scatter plots and 3D interpolated surfaces |
| **Greeks Computation** | Vega calculation for Newton-Raphson convergence |
| **Data Quality Filtering** | Strike range, time-to-expiry, volume, and open interest filters |
| **Advanced Interpolation** | Cubic/linear griddata with Gaussian smoothing |
| **Comprehensive Metrics** | MAE, RMSE, MAPE for price and IV accuracy |

---

##  Methodology

### **Analysis Pipeline**

Data Acquisition → 2. Filtering → 3. IV Computation → 4. Pricing → 5. Validation → 6. Visualization


#### **Step 1: Market Data Retrieval**
- Fetches current spot price and historical data via `yfinance`
- Retrieves options chains for multiple expiration dates
- Extracts bid-ask spreads, volume, open interest, and market-quoted IVs

#### **Step 2: Data Quality Filtering**
- **Strike Range**: 70-130% of spot price (configurable: 85-115% in example)
- **Minimum Time to Expiry**: 25 days (~0.07 years)
- **Price Validation**: Positive last traded prices only
- **Optional Thresholds**: Volume and open interest filters
- **Moneyness Calculation**: K/S ratio and log-moneyness

#### **Step 3: Implied Volatility Calculation**

**Three Computational Methods**:

1. **Newton-Raphson Method**:
   - Fast iterative approach using Vega for gradient descent
   - Initial guess: Brenner-Subrahmanyam approximation for ATM options
   - $$\sigma_{n+1} = \sigma_n - \frac{C_{BS}(\sigma_n) - C_{market}}{Vega(\sigma_n)}$$
   - Max 50 iterations, tolerance: 1e-8
   - Damping for stability when approaching bounds

2. **Brent's Method**:
   - Root-finding algorithm using bracketing
   - Robust fallback when Newton-Raphson fails
   - Dynamic bound adjustment (0.001 to 10.0)
   - Max 150 iterations

3. **Hybrid Method** (Default):
   - Attempts fast Newton-Raphson first
   - Falls back to robust Brent's method if:
     - Newton fails to converge
     - Result outside reasonable bounds (0.1% - 1000%)
     - Verification fails (re-pricing error too large)
   - **Success Rate Tracking**: Reports % of successful IV calculations

**Convergence Parameters**:
- Tolerance: 1e-8
- IV Search Bounds: 0.001 to 10.0 (0.1% to 1000% annualized)
- Minimum Volatility: 1e-10 for numerical stability
- Initial Guess Clipping: 0.05 to 2.0

#### **Step 4: Theoretical Pricing**
- Recalculates option prices using computed IVs
- Generates price error metrics:
  - Absolute error: |Theoretical - Market|
  - Percentage error: (Absolute / Market) × 100
- Compares computed IV vs. market-quoted IV (when available)

#### **Step 5: Accuracy Validation**

**Error Metrics Computed**:

| Metric | Description | Formula |
|--------|-------------|---------|
| **MAE** | Mean Absolute Error | $$\frac{1}{n}\sum\|Theo - Market\|$$ |
| **RMSE** | Root Mean Squared Error | $$\sqrt{\frac{1}{n}\sum(Theo - Market)^2}$$ |
| **MAPE** | Mean Absolute % Error | $$\frac{1}{n}\sum\frac{\|Theo - Market\|}{Market} \times 100$$ |
| **Median Error** | Robust central tendency | Median of absolute errors |

**Segmented Analysis**:
- **All Options**: Complete dataset statistics
- **ATM Options**: Focused analysis within ±12-15% moneyness range
- **IV Comparison**: Computed vs. market-quoted IV errors (when available)

#### **Step 6: Visualization**

**2D Visualizations**:
- **IV Smile by Strike**: Scatter plot showing volatility smile pattern
  - Separate markers for calls vs. puts
  - Overlay market-quoted IV for comparison
  - Spot price reference line
- **IV by Moneyness**: Color-coded by time to expiry
  - Shows term structure effects
  - ATM reference line at K/S = 1.0

**3D Surface Plots**:
- **Computed IV Surface**: Interpolated from calculated IVs
  - X-axis: Time to expiry
  - Y-axis: Moneyness (K/S)
  - Z-axis: Implied Volatility
  - Grid size: 50×50 with Gaussian smoothing (σ=0.7)
- **Market IV Surface**: Based on market-quoted IVs (when available)
- **Difference Surface**: Computed - Market IV
  - Identifies systematic biases
  - Highlights potential arbitrage opportunities
  - Color scale: Red (overpriced) to Blue (underpriced)

**Interpolation Techniques**:
- **Primary**: Cubic griddata interpolation
- **Fallback**: Linear interpolation if cubic fails
- **Smoothing**: Gaussian filter with configurable sigma
- **Outlier Removal**: 3-sigma filter before interpolation
- **Edge Handling**: 5% padding around data range

---

##  Technical Implementation

### **Core Functions**

#### **Black-Scholes Pricing**
```python
black_scholes_price(S, K, r, T, sigma, q=0.0, option_type='call')
black_scholes_vega(S, K, r, T, sigma, q=0.0)
```

- Includes dividend yield adjustment
- Handles edge cases (at expiration, zero volatility)
- Returns intrinsic value on numerical errors

#### **Implied Volatility Calculation**
```python
calculate_iv_newton()     # Newton-Raphson method
calculate_iv_brent()      # Brent's method
calculate_iv()            # Hybrid method (default)
```

#### **Data Pipeline**
```python
get_stock_data()          # Fetch spot price and history
get_options_data()        # Retrieve options chains
filter_options_data()     # Apply quality filters
compute_iv_for_options()  # Batch IV calculation

```

#### **Analysis & Metrics**
```python
calculate_theoretical_prices()  # Reprice with computed IVs
calculate_accuracy_metrics()    # MAE, RMSE, MAPE

```

#### **Visualization**
```python
plot_iv_comparison_2d()   # 2D scatter plots
plot_iv_surfaces_3d()     # 3D surface plots
create_iv_surface_grid()  # Interpolation helper

```

