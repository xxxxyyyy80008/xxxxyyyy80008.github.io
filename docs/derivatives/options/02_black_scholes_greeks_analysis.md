# Black-Scholes Option Pricing with Greeks Analysis


**Notebooks**
- **[Kaggle Notebook](https://www.kaggle.com/code/xxxxyyyy80008/black-scholes-option-pricing-with-greeks-analysis)**
- **[Github Notebook](https://github.com/xxxxyyyy80008/Option-Pricing-and-Risk-Models/blob/main/notebooks/02_black_scholes_greeks_analysis.ipynb)**



##  Overview

This Jupyter notebook provides an **advanced implementation of Black-Scholes-Merton option pricing** with comprehensive **Option Greeks analysis**, featuring multi-dimensional sensitivity analysis, interactive visualizations, Monte Carlo validation, and real market data integration. The notebook demonstrates both theoretical foundations and practical applications for options trading and risk management.

---

##  Objectives

1. **Implement analytical Black-Scholes-Merton pricing formulas** for European options with dividend yield support
2. **Calculate and visualize all five primary Option Greeks** (Delta, Gamma, Vega, Theta, Rho)
3. **Perform multi-dimensional sensitivity analysis** across spot price, volatility, and time parameters
4. **Compare Monte Carlo simulations with analytical solutions** for validation
5. **Analyze real market data** from Yahoo Finance (Boeing stock example)
6. **Demonstrate portfolio-level Greeks aggregation** for risk management

---

##  Components

### **1. Black-Scholes-Merton Pricing Implementation**

**Core Functions**:
- `black_scholes_call()` - European call option pricing
- `black_scholes_put()` - European put option pricing
- Includes dividend yield (q) parameter for comprehensive modeling
- Edge case handling (at-maturity scenarios)

**Formulas Implemented**:

**Call Option**:
$$C = Se^{-q\tau}N(d_1) - Ke^{-r\tau}N(d_2)$$

**Put Option**:
$$P = Ke^{-r\tau}N(-d_2) - Se^{-q\tau}N(-d_1)$$

Where:
$$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)\tau}{\sigma\sqrt{\tau}}$$
$$d_2 = d_1 - \sigma\sqrt{\tau}$$

---

### **2. Option Greeks - Complete Suite**

#### **Greeks Calculation Function**

`black_scholes_greeks()` computes all five primary Greeks:

| Greek | Description | Formula Representation | Units |
|-------|-------------|----------------------|-------|
| **Delta (Δ)** | Sensitivity to spot price | $$\frac{\partial V}{\partial S}$$ | Change per $1 move |
| **Gamma (Γ)** | Rate of change of Delta | $$\frac{\partial^2 V}{\partial S^2}$$ | Delta change per $1 |
| **Vega (ν)** | Sensitivity to volatility | $$\frac{\partial V}{\partial \sigma}$$ | Change per 1% vol |
| **Theta (Θ)** | Time decay | $$\frac{\partial V}{\partial t}$$ | Change per day |
| **Rho (ρ)** | Interest rate sensitivity | $$\frac{\partial V}{\partial r}$$ | Change per 1% rate |

**Key Formulas**:

- **Delta (Call)**: $$\Delta_c = e^{-q\tau}N(d_1)$$
- **Delta (Put)**: $$\Delta_p = e^{-q\tau}(N(d_1) - 1)$$
- **Gamma**: $$\Gamma = \frac{e^{-q\tau}n(d_1)}{S\sigma\sqrt{\tau}}$$
- **Vega**: $$\nu = Se^{-q\tau}\sqrt{\tau}n(d_1) / 100$$
- **Theta (Call)**: $$\Theta_c = -\frac{1}{365}\left[\frac{S\sigma e^{-q\tau}n(d_1)}{2\sqrt{\tau}} + rKe^{-r\tau}N(d_2) - qSe^{-q\tau}N(d_1)\right]$$

---

### **3. Example Calculation - Boeing (BA) Stock**

**Test Parameters**:
- **Spot Price (S)**: $214 (Boeing as of Dec 19, 2025)
- **Strike Price (K)**: $225
- **Volatility (σ)**: 28%
- **Risk-Free Rate (r)**: 3.6%
- **Dividend Yield (q)**: 0%
- **Time to Maturity**: 33 days (0.0904 years)


---

##  Visualization Suite

### **1. Greeks vs Spot Price (5-Panel Line Charts)**

**Comprehensive 1D sensitivity analysis**:
- **Panel 1**: Delta for calls and puts
- **Panel 2**: Gamma (maximum at ATM)
- **Panel 3**: Vega (sensitivity to volatility changes)
- **Panel 4**: Theta (time decay patterns)
- **Panel 5**: Rho (interest rate sensitivity)

**Features**:
- Spot price range: $50 to $350
- Strike reference line at K=$225
- Clear visualization of moneyness effects (ITM/ATM/OTM)
- Delta: 0 to 1 for calls, -1 to 0 for puts
- Gamma: peaked at ATM, demonstrates curvature risk
- Vega: maximum for ATM options

---

### **2. Greeks Heatmaps (2D Contour Plots)**

**4-panel heatmap showing sensitivity to TWO variables simultaneously**:

**Axes**:
- **X-axis**: Spot Price ($80 to $380)
- **Y-axis**: Volatility (10% to 50%)
- **Color intensity**: Greek value

**Panels**:
1. **Delta (Call)**: Shows progression from 0 (deep OTM) to 1 (deep ITM)
2. **Gamma**: Concentrated band around strike price, highest for ATM
3. **Vega**: Maximum in ATM region, demonstrates vol risk concentration
4. **Theta**: Time decay patterns across strike and volatility space

**Color Schemes**:
- Delta: Red-Yellow-Green gradient
- Gamma: Viridis (blue to yellow)
- Vega: Plasma (purple to yellow)
- Theta: Coolwarm (red to blue)

---

### **3. 3D Surface Plots**

**3D surfaces showing Greeks evolution**:

**Two primary surfaces generated**:
- **Delta Surface**: Shows smooth transition from 0 to 1
  - X-axis: Spot Price ($90-$360)
  - Y-axis: Time to Maturity (1-365 days)
  - Z-axis: Delta value
  
- **Gamma Surface**: Shows concentration around strike
  - Peaks at ATM for shorter maturities
  - Flattens as time to expiry increases

**Viewing Parameters**:
- Elevation: 25°
- Azimuth: 45°
- Color mapping: Viridis
- Grid resolution: 30×30


---

##  Advanced Analysis Features

### **4. Greeks Dashboard - Multiple Strikes**

**Tabular analysis across strike prices**:

**Strikes Analyzed**: [170, 195, 205, 215, 225, 255, 280, 295]

**Columns Provided**:
- Strike price
- Moneyness (S/K ratio)
- Status (ITM/ATM/OTM)
- Call & Put prices
- All Greeks for both calls and puts


---

### **5. Monte Carlo Validation**

**Purpose**: Verify analytical formulas through simulation

**Configuration**:
- **Simulation Method**: Geometric Brownian Motion (GBM)
- **Number of Simulations**: 100,000 paths
- **Random Seed**: 3407 (reproducibility)

**GBM Formula**:
$$S_T = S_0 \exp\left[(r - q - \frac{\sigma^2}{2})T + \sigma\sqrt{T}Z\right]$$

Where Z ~ N(0,1)

---

### **6. Real Market Data Integration**

**Data Source**: Yahoo Finance (`yfinance` library)

**Ticker Example**: Boeing (BA)

**Workflow**:
1. **Fetch Current Price**: Real-time spot price retrieval
2. **Get Options Chain**: All available expiration dates
3. **Extract Market Data**: Strikes, last prices, implied volatilities
4. **Calculate Greeks**: Using market-implied volatilities
5. **Compare Theoretical vs Market Prices**

**Risk-Free Rate Curve** (as of Dec 2025):
- Uses U.S. Treasury yield curve
- Interpolates based on time to maturity
- Range: 30 days (3.62%) to 30 years (4.83%)


**Observations**:
- Implied volatility smile visible (higher vol for OTM options)
- Pricing discrepancies due to bid-ask spreads and liquidity
- Greeks provide risk metrics for each strike

---

##  Portfolio Risk Management

### **7. Portfolio Greeks Aggregation**

**Function**: `calculate_portfolio_greeks()`

**Purpose**: Calculate net Greeks exposure for multi-position portfolios

**Example Portfolio**:
```python
Position 1:  +10 x CALL K=$200, σ=28%, T=182 days
Position 2:   -5 x CALL K=$220, σ=28%, T=182 days  # Short call spread
Position 3:   +5 x PUT  K=$180, σ=28%, T=182 days
```

**Aggregate Greeks Output**:

```
  Greek                     Value  Interpretation
  --------------- ---------------  --------------------------------------------------
  Delta                    3.7090  Portfolio moves $3.71 for $1 move in underlying
  Gamma                  0.061512  Delta changes by 0.061512 for $1 move
  Vega                     3.9330  Portfolio moves $3.93 for 1% vol change
  Theta                   -0.3609  Portfolio loses $0.36 per day (time decay)
  Rho                      2.9519  Portfolio moves $2.95 for 1% rate change
``
