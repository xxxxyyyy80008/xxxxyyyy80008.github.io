# Option Pricing: Binomial Tree and Monte Carlo Comparison


**Notebooks**
- **[Github Notebook](https://github.com/xxxxyyyy80008/Option-Pricing-and-Risk-Models/blob/main/notebooks/04_option_pricing_binomial_tree.ipynb)**
- **[Kaggle Notebook](https://www.kaggle.com/code/xxxxyyyy80008/option-pricing-binomial-tree)**


## Overview
Option pricing implementation featuring One-step and two-step binomial trees (CRR and GBM), Monte Carlo simulation with Geometric Brownian Motion and Black-Scholes model for comparison

---

##  Components

### 1. Black-Scholes Functions
- **`black_scholes_d1()`** - Calculate d₁ parameter
- **`black_scholes_d2()`** - Calculate d₂ parameter  
- **`black_scholes_call()`** - European call option price
- **`black_scholes_put()`** - European put option price

**Formula:**

$$
C(S,t) = S e^{-dT} N(d_1) - K e^{-rT} N(d_2)
$$

$$
d_1 = \frac{\ln(S/K) + (r - d + \sigma^2/2)T}{\sigma\sqrt{T}}
$$

$$
d_2 = d_1 - \sigma\sqrt{T}
$$

---

### 2. One-Step Binomial Tree

#### CRR Method
**Function:** `one_step_binomial_crr()`

**Parameters:**

$$
u = e^{\sigma\sqrt{\Delta t}}, \quad d = \frac{1}{u}
$$

$$
p = \frac{e^{(r-d)\Delta t} - d}{u - d}
$$

#### GBM Method
**Function:** `one_step_binomial_gbm()`

**Parameters:**

$$
u = e^{(r-d-0.5\sigma^2)\Delta t + \sigma\sqrt{\Delta t}}
$$

$$
d = e^{(r-d-0.5\sigma^2)\Delta t - \sigma\sqrt{\Delta t}}
$$

**Returns:**
- Option price at t=0
- Stock price tree
- Option value tree
- Risk-neutral probability
- Up/down multipliers

---

### 3. Two-Step Binomial Tree

**Function:** `two_step_binomial_crr()`


**Features:**

- European and American options
- Early exercise capability
- Backward induction pricing
- Visual tree construction


---

### 4. N-Step Binomial Tree

**Function:** `n_step_binomial_crr()`

**Convergence:**
As N → ∞, binomial price → Black-Scholes price

**Typical Values:**
- N = 50: Good approximation
- N = 100: Excellent accuracy
- N = 200+: Near-perfect convergence

**Complexity:** O(N²) time, O(N²) space

---

### 5. Monte Carlo Simulation (GBM)

**Function:** `monte_carlo_gbm()`

**Geometric Brownian Motion:**

$$
S_t = S_0 \exp\left[(r - d - \frac{\sigma^2}{2})t + \sigma W_t\right]
$$

**Discretization:**

$$
S_{t+\Delta t} = S_t \exp\left[(r - d - \frac{\sigma^2}{2})\Delta t + \sigma\sqrt{\Delta t}Z\right]
$$

where Z ~ N(0,1)

**Output:**
- Option price estimate
- Standard error
- Simulated price paths

**Convergence Rate:** O(1/√M) where M = number of simulations

---


##  Visualizations 

### 1. Convergence Analysis
- **Binomial Tree Convergence**: N-step prices vs Black-Scholes
- **Monte Carlo Convergence**: Price and standard error vs simulations

### 2. Monte Carlo Paths
- 20 sample paths from 1,000 simulations
- Strike price reference line
- Time evolution from 0 to T

### 3. Two-Step Tree Diagrams
Side-by-side visualization:
- Stock price tree with values
- Option value tree with values
- Arrow connections between nodes

---


## Features

###  Numerical Methods
1. **Binomial Trees**: CRR and GBM parameterizations
2. **Monte Carlo**: Geometric Brownian Motion simulation
4. **Backward Induction**: American option pricing

###  Option Types Supported
- European Call/Put
- American Call/Put
- Binary Call/Put (cash-or-nothing)
- Forward Contracts

---

##  Mathematical Foundations

### Risk-Neutral Valuation

$$
C_0 = e^{-rT} \mathbb{E}^Q[\max(S_T - K, 0)]
$$

### Binomial Risk-Neutral Probability

$$
p = \frac{e^{(r-d)\Delta t} - d}{u - d}
$$

### No-Arbitrage Condition

$$
d < e^{r\Delta t} < u
$$

### Put-Call Parity

$$
C - P = S_0 e^{-dT} - K e^{-rT}
$$

---

##  Usage Example

```python
# Set parameters
S0 = 100.0    # Current stock price
K = 100.0     # Strike price
r = 0.05      # Risk-free rate (5%)
d = 0.02      # Dividend yield (2%)
sigma = 0.2   # Volatility (20%)
T = 1.0       # Time to maturity (1 year)

# Black-Scholes price
bs_call = black_scholes_call(S0, K, r, d, sigma, T)

# One-step binomial
one_step_price, _, _, _, _, _ = one_step_binomial_crr(S0, K, r, d, sigma, T, 'call')

# Two-step binomial
two_step_price, stock_tree, option_tree, _, _, _ = two_step_binomial_crr(
    S0, K, r, d, sigma, T, 'call'
)

# N-step convergence
n100_price, _, _ = n_step_binomial_crr(S0, K, r, d, sigma, T, 100, 'call')

# Monte Carlo
mc_price, mc_error, paths = monte_carlo_gbm(
    S0, K, r, d, sigma, T, 10000, 100, 'call', seed=42
)


```

