#  Delta Hedging 


**Notebooks**
- **[Kaggle Notebook](https://www.kaggle.com/code/xxxxyyyy80008/option-pricing-delta-hedging)**
- **[Github Notebook](https://github.com/xxxxyyyy80008/Option-Pricing-and-Risk-Models/blob/main/notebooks/05_delta_hedging.ipynb)**


## Overview
This notebook implements **delta hedging strategies** for European options, with detailed profit & loss (P&L) attribution and hedging error analysis. 

---

## Features

### 1. **Dynamic Delta Hedging Simulation**
- Implements self-financing delta-neutral hedging strategy
- Simulates realistic stock price paths using Geometric Brownian Motion (GBM)
- Periodic rebalancing with configurable frequency (daily, weekly, etc.)
- Incorporates transaction costs to reflect real-world trading conditions

### 2. **Detailed P&L Attribution**
Decomposes hedging performance into four key components:

| Component | Description | Impact |
|-----------|-------------|---------|
| **Delta P&L** | First-order hedging effect | Linear with stock price changes |
| **Gamma P&L** | Hedging error from convexity | Captures second-order effects |
| **Theta P&L** | Time decay contribution | Reflects option's time value erosion |
| **Transaction Costs** | Rebalancing friction | Real-world implementation cost |

### 3. **Hedging Error Analysis**
- Quantifies relationship between rebalancing frequency and hedging accuracy
- Demonstrates the trade-off between hedging precision and transaction costs
- Statistical analysis across multiple market scenarios (100+ simulations)

---

## Technical Implementation

### Core Components
1. **Black-Scholes Framework**: Option valuation and Greek calculations
2. **Monte Carlo Engine**: Stochastic stock price simulation
3. **Portfolio Tracking**: Real-time hedge ratio adjustment
4. **Statistical Analysis**: Multi-scenario performance assessment

### Visualization Suite
- Stock price path and delta evolution
- Cumulative P&L attribution by component
- Hedging error vs. rebalancing frequency
- Distribution analysis of hedging outcomes


---


## Configuration

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `S0` | float | 100.0 | Initial stock price |
| `K` | float | 100.0 | Strike price |
| `r` | float | 0.05 | Risk-free interest rate (annual) |
| `d` | float | 0.02 | Dividend yield (annual) |
| `sigma` | float | 0.2 | Volatility (annual) |
| `T` | float | 1.0 | Time to maturity (years) |
| `num_steps` | int | 50 | Number of rebalancing steps |
| `option_type` | str | 'call' | 'call' or 'put' |
| `transaction_cost` | float | 0.001 | Transaction cost (fraction) |
| `seed` | int | None | Random seed for reproducibility |

### Customization

```python
# Custom parameter set for high volatility scenario
high_vol_params = {
    'S0': 100.0,
    'K': 100.0,
    'r': 0.05,
    'd': 0.02,
    'sigma': 0.4,      # 40% volatility
    'T': 0.5,          # 6 months
    'num_steps': 100,  # Daily rebalancing
    'option_type': 'call',
    'transaction_cost': 0.002,  # 0.2% costs
    'seed': 123
}

results = delta_hedging_simulation(**high_vol_params)
```

##  Theory 

### Black-Scholes Model

The framework uses the Black-Scholes-Merton model for option pricing:

$$
C(S,t) = S e^{-dT} N(d_1) - K e^{-rT} N(d_2)
$$

Where:

$$
d_1 = \frac{\ln(S/K) + (r - d + \sigma^2/2)T}{\sigma\sqrt{T}}
$$

$$
d_2 = d_1 - \sigma\sqrt{T}
$$

### Delta Hedging Strategy

**Objective**: Construct a portfolio that replicates the option payoff

**Portfolio Construction**:
- **Short 1 option** (liability)
- **Long Δ shares** (hedge)
- **Cash account** (financing)

**Rebalancing Rule**:

$$
\Delta_t = \frac{\partial C}{\partial S} = e^{-dT} N(d_1)
$$

### P&L Attribution

**Total P&L Decomposition**:


$$
\text{PnL} = \Delta \cdot dS + \frac{1}{2}\Gamma \cdot (dS)^2 + \Theta \cdot dt - \text{Costs}
$$

- **Delta P&L**: Linear hedge performance
- **Gamma P&L**: Convexity error (main source of hedging risk)
- **Theta P&L**: Time value decay
- **Transaction Costs**: Friction from rebalancing

### Hedging Error

Discrete rebalancing introduces error that scales as:

$$
\text{Error} \sim \sigma \sqrt{\Delta t}
$$

**Implication**: More frequent rebalancing reduces error but increases costs.
