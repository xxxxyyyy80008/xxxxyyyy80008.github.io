# Portfolio Optimization using Monte Carlo Methods

**Notebooks**

- [Github Notebook](https://github.com/xxxxyyyy80008/Quantitative-Portfolio-Optimization/blob/main/notebooks/01_portfolio_optimization_monte_carlo.ipynb)
- [Kaggle Notebook](https://www.kaggle.com/code/xxxxyyyy80008/portfolio-optimization-with-monte-carlo)


##  Overview
This notebook implements **Modern Portfolio Theory (MPT)** combined with **Monte Carlo simulation** to construct an optimal investment portfolio. The analysis covers a 7-year period from January 2019 to December 2025, utilizing 9 stocks across various sectors.

##  Objective
Maximize risk-adjusted returns (Sharpe Ratio) through optimal portfolio allocation while respecting diversification constraints.

---

##  Methodology

### 1. **Data Collection & Processing**
- **Data Source**: Yahoo Finance API (yfinance)
- **Stock Universe**: 9 stocks (AEP, TPL, BALL, A, SO, TJX, UNH, AXON, JPM)
- **Time Period**: January 1, 2019 - January 1, 2026
- **Data Type**: Daily adjusted close prices
- **Returns Calculation**: Logarithmic returns using $$R_i = \log(r_i / r_{i-1})$$

### 2. **Statistical Analysis**
- **Annualized Returns**: $$\mu^{Annual} = 252 \times \mu$$
- **Annualized Covariance**: $$\sigma_{ij}^{Annual} = 252 \times \sigma_{ij}$$
- **Annualized Volatility**: $$\sigma^{Annual} = \sqrt{252} \times \sigma$$
- **Correlation Matrix**: Computed to assess diversification benefits

### 3. **Monte Carlo Simulation**
- **Iterations**: 50,000 random portfolio combinations
- **Weight Generation**: Random weights $$W_i$$ where $$\sum W_i = 1$$
- **Portfolio Return**: $$\mu_P^{Annual} = \sum W_i \mu_i^{Annual}$$
- **Portfolio Volatility**: $$\sigma_P^{Annual} = \sqrt{\sum_{ij} W_i W_j \sigma_{ij}^{Annual}}$$
- **Purpose**: Map the efficient frontier across risk-return spectrum

### 4. **Portfolio Optimization**
- **Objective Function**: Maximize Sharpe Ratio
- **Sharpe Ratio**: $$(R_p - R_f) / \sigma_p$$
- **Risk-Free Rate**: 3.6% (1-year US Treasury yield as of Dec 20, 2025)
- **Algorithm**: SLSQP (Sequential Least Squares Programming)
- **Constraints**:
  - Portfolio weights sum to 100%
  - No short selling: $$0 \leq W_i \leq 1$$

---

##  Technical Implementation

### Libraries Used
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **yfinance**: Stock data retrieval
- **SciPy**: Optimization algorithms

### Key Functions
1. `download_stock_data()` - Historical price data retrieval
2. `calculate_returns()` - Compute daily and logarithmic returns
3. `initialize_weights()` - Generate random portfolio weights
4. `portfolio_statistics()` - Calculate return, volatility, and Sharpe ratio
5. `generate_portfolios()` - Monte Carlo simulation engine
6. `optimize_portfolio()` - SLSQP optimization wrapper

### Configuration Parameters
- **Trading Days**: 252 per year
- **Random Seed**: 3407 (for reproducibility)
- **Monte Carlo Portfolios**: 50,000
- **Risk-Free Rate**: 0.036 (3.6%)

---

##  Key Results & Visualizations

### 1. **Data Visualization**
- Stock price evolution over time
- Daily logarithmic returns time series
- Distribution histograms of returns

### 2. **Statistical Analysis**
- Annualized returns for each stock
- Annualized volatility metrics
- Correlation heatmap showing inter-stock relationships

### 3. **Efficient Frontier**
- 50,000 simulated portfolios plotted on risk-return space
- Color-coded by Sharpe Ratio (viridis colormap)
- Optimal portfolio marked with red star
- Clear visualization of risk-return trade-off

### 4. **Optimal Portfolio**
- **Allocation**: Pie chart and bar chart showing optimal weights
- **Performance Metrics**:
  - Expected Annual Return
  - Expected Annual Volatility
  - Sharpe Ratio

### 5. **Summary Statistics**
Comprehensive table including:
- Individual stock optimal weights
- Annual returns per stock
- Annual volatility per stock
- Individual Sharpe ratios

---

##  Key Insights

### Diversification Benefits
- Correlation matrix reveals varying relationships between stocks
- Portfolio volatility reduced through diversification
- Multi-sector exposure (utilities, retail, healthcare, finance, technology)

### Risk-Return Trade-off
- Monte Carlo simulation clearly maps efficient frontier
- Optimal portfolio positioned on efficient frontier
- Maximum risk-adjusted return achieved for given volatility level

### Optimization Success
- SLSQP algorithm successfully converges
- Constraints satisfied (weights sum to 1, no short positions)
- Optimal allocation maximizes Sharpe Ratio

---

##  Output Deliverables

1. **Visualizations**:
   - Stock price evolution chart
   - Returns distribution histograms
   - Correlation heatmap
   - Efficient frontier scatter plot
   - Portfolio allocation pie chart
   - Portfolio weights bar chart

2. **Statistical Reports**:
   - Annualized returns table
   - Volatility metrics
   - Covariance matrix
   - Optimal portfolio allocation
   - Performance metrics summary

3. **Optimal Portfolio**:
   - Individual stock weights
   - Expected return and volatility
   - Sharpe Ratio
   - Complete performance summary

---

##  Reference
GitHub Repository: [Monte-Carlo-Portfolio-Allocation](https://github.com/aldodec/Monte-Carlo-Portfolio-Allocation/tree/master)

---

