# Portfolio Optimization using Monte Carlo Methods & PyPortfolioOpt

**Notebooks**

- [Github Notebook](https://github.com/xxxxyyyy80008/Quantitative-Portfolio-Optimization/blob/main/notebooks/02_portfolio_optimization_pyportfolioopt.ipynb)
- [Kaggle Notebook](https://www.kaggle.com/code/xxxxyyyy80008/portfolio-optimization-mc-and-pyportfolioopt)


##  Overview
This notebook demonstrates **dual optimization approaches** for portfolio construction, combining **Modern Portfolio Theory (MPT)**, **Monte Carlo simulation**, and both **traditional (SciPy)** and **advanced (PyPortfolioOpt)** optimization techniques. The analysis spans 7 years from January 2019 to December 2025 across 9 diversified stocks.

##  Objective
Construct optimal investment portfolios using multiple optimization strategies:
1. **Traditional Optimization**: SciPy SLSQP (Sequential Least Squares Programming)
2. **Advanced Optimization**: PyPortfolioOpt library with multiple strategies
3. **Practical Implementation**: Discrete allocation for real-world share purchases

---

##  Methodology

### 1. **Data Collection & Processing**
- **Data Source**: Yahoo Finance API (yfinance)
- **Stock Universe**: 9 stocks (AEP, TPL, BALL, A, SO, TJX, UNH, AXON, JPM)
- **Time Period**: January 1, 2019 - January 1, 2026
- **Returns Calculation**: 
  - Simple daily returns: $$(r_t - r_{t-1})/r_{t-1}$$
  - Logarithmic returns: $$R_i = \log(r_i / r_{i-1})$$

### 2. **Statistical Analysis**
- **Annualized Returns**: $$\mu_p = \sum_{i=1}^{n} w_i \mu_i \times 252$$
- **Portfolio Volatility**: $$\sigma_p = \sqrt{w^T \Sigma w \times 252}$$
- **Sharpe Ratio**: $$SR = \frac{\mu_p - r_f}{\sigma_p}$$
- **Correlation Matrix**: Computed for diversification analysis

### 3. **Monte Carlo Simulation**
- **Iterations**: 50,000 random portfolio combinations
- **Purpose**: Map efficient frontier across risk-return spectrum
- **Random Weight Generation**: $$W_i$$ where $$\sum W_i = 1$$
- **Visualization**: Color-coded by Sharpe Ratio using viridis colormap

### 4. **Dual Portfolio Optimization Approach**

#### 4.1 **Traditional Optimization (SciPy SLSQP)**
- **Algorithm**: Sequential Least Squares Programming
- **Objective**: Maximize Sharpe Ratio (minimize negative Sharpe)
- **Constraints**:
  - Portfolio weights sum to 100%: $$\sum w_i = 1$$
  - No short selling: $$0 \leq w_i \leq 1$$
- **Method**: Gradient-based optimization

#### 4.2 **Advanced Optimization (PyPortfolioOpt)**
Three optimization strategies implemented:

1. **Maximum Sharpe Ratio**
   - Objective: Optimal risk-adjusted returns
   - Uses mean historical returns and sample covariance

2. **Minimum Volatility**
   - Objective: Conservative risk minimization
   - Focuses on lowest portfolio variance

3. **Efficient Risk** (Target Volatility)
   - Objective: Target-volatility portfolio construction
   - Customizable risk tolerance (e.g., 20% volatility)

#### 4.3 **Discrete Allocation**
- **Portfolio Value**: $100,000
- **Algorithm**: Greedy allocation maximizing portfolio value utilization
- **Output**: Actual number of shares to purchase per stock
- **Practical Feature**: Calculates leftover cash after allocation

---

##  Technical Implementation

### Core Libraries
- **NumPy/Pandas**: Numerical computation and data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **yfinance**: Historical stock data retrieval
- **SciPy**: Traditional optimization (SLSQP algorithm)
- **PyPortfolioOpt**: Advanced portfolio optimization
  - `EfficientFrontier`: Core optimization class
  - `risk_models`: Covariance estimation
  - `expected_returns`: Return forecasting
  - `DiscreteAllocation`: Share quantity calculation

### Key Functions

#### Traditional Approach (SciPy)
1. `initialize_weights()` - Generate random portfolio weights
2. `calculate_portfolio_return()` - Compute annualized returns
3. `calculate_portfolio_volatility()` - Calculate portfolio risk
4. `calculate_sharpe_ratio()` - Risk-adjusted performance metric
5. `negative_sharpe_ratio()` - Objective function for minimization
6. `optimize_portfolio()` - SLSQP wrapper function

#### Advanced Approach (PyPortfolioOpt)
1. `optimize_portfolio_pypfopt()` - Multi-method optimization wrapper
2. `calculate_discrete_allocation()` - Convert weights to share quantities
3. `plot_pypfopt_efficient_frontier()` - Visualize efficient frontier curve

### Configuration Parameters
- **Trading Days**: 252 per year
- **Random Seed**: 3407 (reproducibility)
- **Monte Carlo Portfolios**: 50,000
- **Risk-Free Rate**: 0.036 (3.6% - 1yr US Treasury yield, Dec 20, 2025)
- **Portfolio Value**: $100,000 (for discrete allocation)
- **Covariance Estimation**: Sample covariance with 252-day annualization

---

##  Key Results & Visualizations

### 1. **Data Exploration**
- Stock price evolution time series
- Daily logarithmic returns visualization
- Return distribution histograms (150 bins)
- Correlation heatmap with blue color scheme

### 2. **Statistical Analysis**
- Annualized returns per stock
- Annualized volatility metrics
- Correlation matrix revealing diversification opportunities
- Covariance matrix (annualized)

### 3. **Monte Carlo Efficient Frontier**
- 50,000 simulated portfolios plotted
- Risk-return trade-off visualization
- Color gradient showing Sharpe Ratio spectrum
- Optimal portfolios marked:
  - **Red star**: SciPy SLSQP optimal portfolio
  - **Lime diamond**: PyPortfolioOpt optimal portfolio

### 4. **Optimization Results**

#### **SciPy SLSQP Results**
- Optimal portfolio weights (% allocation per stock)
- Expected annual return
- Expected annual volatility
- Sharpe Ratio

#### **PyPortfolioOpt Max Sharpe Results**
- Cleaned weights (removes insignificant positions)
- Expected annual return
- Expected annual volatility
- Sharpe Ratio

#### **PyPortfolioOpt Min Volatility Results**
- Conservative weight allocation
- Minimized portfolio risk
- Expected return and Sharpe Ratio


### 6. **Comparative Analysis**
Comparison table showing:
- **Method**: SciPy SLSQP vs PyPortfolioOpt (Max Sharpe) vs PyPortfolioOpt (Min Volatility)
- **Expected Return**: Annualized percentage
- **Volatility**: Annualized risk measure
- **Sharpe Ratio**: Risk-adjusted performance

### 7. **Visualization Suite**

#### Portfolio Weight Visualizations
1. **Pie Charts**: 
   - SciPy optimal allocation
   - PyPortfolioOpt optimal allocation (filtered for non-zero weights)

2. **Bar Charts**:
   - SciPy weights with value labels
   - Side-by-side comparison (SciPy vs PyPortfolioOpt)

3. **Efficient Frontier Curves**:
   - Monte Carlo scatter plot with both optimal portfolios
   - PyPortfolioOpt efficient frontier curve with:
     - Individual asset positions
     - Max Sharpe portfolio (red star)
     - Min Volatility portfolio (lime diamond)

### 8. **Detailed Summary Statistics**
Comprehensive table including:
- Stock ticker
- SciPy optimal weight (5 decimal precision)
- PyPortfolioOpt optimal weight
- Individual stock annual return
- Individual stock annual volatility
- Individual stock Sharpe Ratio

---

##  Key Insights

### Dual Optimization Comparison
- **SciPy SLSQP**: 
  - Traditional gradient-based approach
  - Fast convergence
  - Suitable for standard constraints
  
- **PyPortfolioOpt**:
  - Modern library with multiple strategies
  - Built-in weight cleaning
  - Discrete allocation functionality
  - Multiple optimization objectives available

### Diversification Benefits
- Correlation matrix reveals varying inter-stock relationships
- Multi-sector exposure (utilities, retail, healthcare, finance, technology)
- Portfolio volatility reduced through diversification
- Optimal portfolios positioned on efficient frontier

### Practical Implementation
- Discrete allocation bridges theory and practice
- Converts continuous weights to integer share quantities
- Maximizes portfolio value utilization
- Calculates remaining cash for liquidity

### Risk-Return Trade-offs
- Maximum Sharpe: Best risk-adjusted returns
- Minimum Volatility: Conservative approach with lower risk
- Clear visualization of efficient frontier
- Multiple optimal portfolios for different investor preferences

---

##  Output Deliverables

1. **Visualizations** (10+ charts):
   - Stock price evolution
   - Returns distribution histograms
   - Correlation heatmap
   - Monte Carlo efficient frontier (with dual optimal portfolios)
   - PyPortfolioOpt efficient frontier curve
   - Pie charts (2 methods)
   - Bar charts (individual and comparative)
   - Weight comparison charts

2. **Statistical Reports**:
   - Annualized returns and volatilities
   - Covariance and correlation matrices
   - Optimal portfolio allocations (2 methods)
   - Performance metrics comparison
   - Detailed summary statistics

3. **Practical Outputs**:
   - Discrete share allocation
   - Exact number of shares per stock
   - Investment amounts per position
   - Remaining cash calculation

---

##  Technical Advantages

### PyPortfolioOpt Benefits
- **Flexibility**: Multiple optimization methods
- **Robustness**: Built-in weight cleaning and validation
- **Practicality**: Discrete allocation for real trading
- **Extensibility**: Easy to add custom objectives
- **Visualization**: Built-in plotting capabilities

### Comparison Framework
- Side-by-side method comparison
- Consistent performance metrics
- Visual weight comparisons
- Comprehensive summary tables

---

##  References
- **GitHub Repository**: [Monte-Carlo-Portfolio-Allocation](https://github.com/aldodec/Monte-Carlo-Portfolio-Allocation/tree/master)
- **PyPortfolioOpt Documentation**: [https://pyportfolioopt.readthedocs.io/en/latest/](https://pyportfolioopt.readthedocs.io/en/latest/)

---
