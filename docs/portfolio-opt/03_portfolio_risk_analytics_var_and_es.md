# Portfolio Risk Analytics: VaR and ES Estimation

**Notebooks**

- [Github Notebook](https://github.com/xxxxyyyy80008/Quantitative-Portfolio-Optimization/blob/main/notebooks/03_portfolio_risk_analytics_var_and_es.ipynb)
- [Kaggle Notebook](https://www.kaggle.com/code/xxxxyyyy80008/portfolio-risk-analytics-var-and-es)

##  Project Overview
This notebook implements a **comprehensive risk analytics framework** featuring **17 distinct Value-at-Risk (VaR) and Expected Shortfall (ES) methodologies** for portfolio risk measurement. The analysis provides a comparative study of traditional, advanced, and modern risk estimation techniques.

##  Objective
Evaluate and compare multiple VaR/ES estimation methodologies to identify the most accurate and robust approaches for portfolio risk measurement, with validation through rigorous backtesting.

---

##  Analysis Specifications

### Configuration Parameters
- **Analysis Period**: January 1, 2019 - December 22, 2025
- **Confidence Level**: 99% (α = 0.01)
- **Rolling Window**: 250 days (~1 trading year)
- **Portfolio**: 11 stocks (ADBE, BALL, DASH, DUK, IFF, SO, TJX, UNH, AXON, JPM, MAR)
- **Random Seed**: 3407 (reproducibility)

### Risk Parameters
- **EWMA Lambda**: 0.94 (exponential decay)
- **Bootstrap Samples**: 1,000 iterations
- **Bootstrap Block Size**: 5
- **Volatility Window**: 20 days
- **Age-Weighted Decay**: 0.98
- **Correlation Window**: 60 days

---

##  Methodology: 17 VaR/ES Methods

### 1. **Parametric Methods** (6 methods)

#### 1.1 Normal Distribution VaR/ES
- **Assumption**: Gaussian return distribution
- **Formula**: $$VaR = -(\mu + z_\alpha \sigma)$$
- **ES Formula**: $$ES = -(\mu - \sigma \frac{\phi(z_\alpha)}{\alpha})$$
- **Advantage**: Simple, closed-form solution
- **Limitation**: Underestimates tail risk

#### 1.2 Student's t-Distribution VaR/ES
- **Assumption**: Heavy-tailed distribution
- **Method**: Maximum likelihood estimation of degrees of freedom
- **Formula**: $$VaR = -(loc + t_\alpha(df) \times scale)$$
- **Advantage**: Better captures fat-tail events
- **Application**: More realistic for financial returns

#### 1.3 EWMA (Exponentially Weighted Moving Average) VaR
- **Decay Factor**: λ = 0.94
- **Variance Formula**: $$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_t^2$$
- **Advantage**: Adapts to changing volatility
- **Use Case**: Trending markets

#### 1.4 Cornish-Fisher VaR
- **Method**: Moment expansion accounting for skewness and kurtosis
- **Adjustment**: $$z_{CF} = z + \frac{(z^2-1)S}{6} + \frac{(z^3-3z)K}{24} - \frac{(2z^3-5z)S^2}{36}$$
- **Advantage**: Non-normal distribution accommodation
- **Parameters**: Incorporates 3rd and 4th moments

---

### 2. **Non-Parametric Methods** (4 methods)

#### 2.1 Historical Simulation VaR/ES
- **Method**: Empirical quantile estimation
- **VaR**: $$VaR = -\text{percentile}(returns, \alpha \times 100)$$
- **ES**: Mean of tail losses beyond VaR
- **Advantage**: No distributional assumptions
- **Limitation**: Limited by historical data availability

#### 2.2 Kernel Density Estimation (KDE) VaR/ES
- **Method**: Smoothed continuous density estimation
- **Bandwidth**: Scott's rule
- **Grid Size**: 10,000 points
- **Advantage**: Smooth distribution estimate
- **Application**: Better tail interpolation than raw historical

---

### 3. **Advanced Historical Simulation Methods** (10 methods)

#### 3.1 Bootstrapped Historical Simulation (BHS) VaR/ES
- **Technique**: Block bootstrap resampling
- **Parameters**:
  - 1,000 bootstrap samples
  - Block size: 5 observations
  - Median of bootstrap estimates
- **Advantage**: Confidence intervals for VaR
- **Method**: Preserves temporal dependencies

#### 3.2 Age-Weighted Historical Simulation (AWHS) VaR/ES
- **Weight Formula**: $$w_i = \lambda^{n-i}$$ (exponential decay)
- **Decay Factor**: λ = 0.98
- **Normalization**: Weights sum to 1
- **Advantage**: Recent observations more influential
- **Application**: Adapts to regime changes

#### 3.3 Volatility-Weighted Historical Simulation (VWHS) VaR/ES
- **Method**: Scale historical returns by current volatility regime
- **Formula**: $$r_{scaled} = r_{hist} \times \frac{\sigma_{current}}{\sigma_{hist}}$$
- **Volatility Window**: 20 days
- **Advantage**: Adjusts for volatility clustering
- **Use Case**: Changing market conditions

#### 3.4 Correlation-Weighted Historical Simulation (CWHS) VaR/ES
- **Method**: Weight scenarios by correlation similarity to current regime
- **Distance Metric**: Frobenius norm of correlation matrices
- **Weight Formula**: $$w_i = \exp(-\frac{d_i}{2})$$
- **Correlation Window**: 60 days
- **Advantage**: Captures regime-dependent correlations
- **Application**: Multi-asset portfolios

#### 3.5 Filtered Historical Simulation (FHS) VaR/ES
- **Method**: Volatility-adjusted historical scenarios
- **Volatility Estimation**: Optional GARCH(1,1) or rolling window
- **Filter Window**: 30 days
- **Scaling**: Current vol / historical vol
- **Advantage**: Combines parametric volatility with non-parametric shocks
- **Fallback**: Simple volatility scaling if GARCH unavailable

---

##  Technical Implementation

### Core Libraries
- **NumPy/Pandas**: Numerical computation and data handling
- **yfinance**: Historical stock price data
- **SciPy**: Statistical distributions and optimization
  - `norm`, `t`, `chi2`: Distribution functions
  - `gaussian_kde`: Kernel density estimation
- **Matplotlib**: Visualization
- **arch**: GARCH modeling (optional for FHS)

### Key Functions

#### Data Processing
1. `download_price_data()` - Yahoo Finance data retrieval with robust error handling
2. `construct_portfolio()` - Build portfolio from notional amounts
   - Calculate shares from notional and latest prices
   - Compute position values, P&L, and returns
   - Handle zero prices gracefully

#### VaR/ES Calculation Functions
1. **Parametric**:
   - `parametric_var_normal()` / `parametric_es_normal()`
   - `parametric_var_student_t()` / `parametric_es_student_t()`
   - `ewma_var()` with rolling EWMA variance calculation
   - `cornish_fisher_var()` with moment adjustments

2. **Non-Parametric**:
   - `historical_var()` / `historical_es()`
   - `kde_historical_var()` / `kde_historical_es()`

3. **Advanced Historical**:
   - `bootstrapped_historical_var()` / `bootstrapped_historical_es()`
   - `age_weighted_historical_var()` / `age_weighted_historical_es()`
   - `volatility_weighted_historical_var()` / `volatility_weighted_historical_es()`
   - `correlation_weighted_historical_var()` / `correlation_weighted_historical_es()`
   - `filtered_historical_var_enhanced()` / `filtered_historical_es_enhanced()`

#### Rolling Computation
- `compute_comprehensive_rolling_var()` - Master function computing all 17 methods
  - Handles stochastic methods with seed control
  - Robust error handling for each method
  - Progress tracking (17 steps)
  - Multi-asset correlation-weighted calculations


---

