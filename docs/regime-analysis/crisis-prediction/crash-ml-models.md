---
layout: default
title: "Equity Market Crisis Regime Prediction"
parent: Crisis Prediction
grand_parent: Regime Analysis
has_children: false
has_toc: false
permalink: /docs/regime-analysis/crisis-prediction
nav_order: 2
---

# Equity Market Crisis Regime Prediction using Machine Learning GBDT
{: .fs-7 }


## Predict Stock Market Crashes

This series is based on the following papers:

- Benhamou, Eric and Ohana, Jean-Jacques and Saltiel, David and Guez, Beatrice, *Planning in Financial Markets in Presence of Spikes: Using Machine Learning GBDT* (June 8, 2021). Université Paris-Dauphine Research Paper No. 3862428. Available at [SSRN](https://ssrn.com/abstract=3862428)  
- Benhamou, Eric and Saltiel, David and Tabachnik, Serge and Bourdeix, Corentin and Chareyron, François and Guez, Beatrice, *Adaptive Supervised Learning for Volatility Targeting Models (Ecml Pkdd Midas 2021 Presentation Slides)* (September 18, 2021). Available at [SSRN](https://ssrn.com/abstract=3926218)
- Ungari, Sandrine and Benhamou, Eric, *Deep Reinforcement Learning for Portfolio Allocation* (July 14, 2021). Risk Magazine Global Quant Network 2021. Available at [SSRN](https://ssrn.com/abstract=3886804)

---

## Define the target variable

Assume there are two regimes for equity markets:

- a **normal regime** where an asset manager should be long to benefit from the long bias of equity markets.
- a **crisis regime**, where an asset manager should either reduce its equity exposure or even sell short it if the strategy is a long short one.

Binary classification target:

- **crisis regime:** if returns (of S&P 500) are below the historical 5 percentile computed on the training data set. (encoded as 1)
- **normal regime:** encoded as 0

---

## Initial (raw) data and feature engineering

The ~150 data series grouped to the following categories:

- The **Risk Aversion** metrics include the equities’ and G10/emerging currencies’ implied volatilities, the High Yield corporate credit bonds credit spreads, and the shape of the VIX forward curve, defined as the ratio of the VIX Spot over the VIX three-month forward. These indicators characterize the financial assets’ liquidity conditions or the accessibility of funding, two complementary measures of risk appetite.
- **Financial** metrics include the one month, six months and one year growth of Earnings per Share, Price/Earnings and Price/Sales for each equity index. These indicators predict the earnings and sales growth cycle, while providing an insight into valuation multiples changes.
- **Macroeconomic** indicators consist of the Citigroup Economic Surprise indices in the main economic zones (US,EU, Japan, Emerging, Worldwide). These indicators convey the cycle of positive or negative economic surprises on a daily basis.
- **US Yields change** (10 years yield, 2 years yield, 10 year breakeven, US Libor) over the same horizons: one month, six months and one year. A change in yields may either reflect the business cycle, the inflation cycle, or the monetary stance of the Federal Reserve.
- The **steepness of the US yield curve** is also computed as a difference between the government bond yield rate and the short term LIBOR rate on two distinct maturities (10 years, 2 years). This indicator is a well-known predictor of the economic cycle as it computes the spread between long term and short term rates.
- **Technical** indicators comprise the put/call ratio (as provided by the CBOE), and the market breadth (the percentage of individual stocks above their respective 200 days Moving Average) on the six equity indices and the MSCI World ACWI. The Put/Call ratio may reflect extreme optimism or pessimism in the investors’ consensus while market breadth characterizes the unweighted average participation of individual stocks among the global equity indices.
- Technical indicators from **various asset classes** are analyzed:
  - Excess returns of six equity indices, BCOM Energy and Industrial Metals, FX Emerging Bloomberg Index Excess Return (reflecting the aggregate evolution of 8 emerging currencies vs. the dollar), dollar index, as computed by the ICE US. Returns are computed over the same time horizons as before (one month, six months and one year),
  - Historical volatilities, computed over horizons of 10,20 and 30 days,
  - Distance to 250 days and 500 days moving average.
  - Sharpe Ratios of all the above-mentioned assets, evaluated over horizons of 6 months and 1 year.
- Cyclical commodities, the dollar index as well as emerging currencies are often leading indicators of the economic cycle. Furthermore, cyclical asset returns and volatilities may either be used procyclically or countercyclically to predict an incoming crisis.

### 102 features are engineered upon the 150+ data series

- 102 features for each of the 150+ series, making 10k+ features to fit into the feature selection process
- These features are used to predict the crash probability in the equity markets.
- These features capture the universal behaviors documented in (Kahneman 2011), namely herding and trending behavior, cross-market contagions, leverage procyclicality etc.
- They also contained a mix of fundamental and technical indicators to capture the two main approaches used in the asset management industry.

---

## Data used in this project

It is impossible to fully obtain the datasets described in the papers via public available dataset. So for this project:

- market data is downloaded from yahoo using package **yfinance**
- economic data is downloaded from **fred.stlouisfed.org**

```markdown
# Equity Market Crisis Regime Prediction using Machine Learning GBDT

## Part I: Introduction

This repository/notebook series builds an **equity crisis regime classifier** using **gradient-boosted decision trees (GBDT)**, with a data pipeline that combines:

- **Yahoo Finance** market data (via `yfinance`)
- **FRED** macro/financial conditions series (via `fredapi`)
- A reproducible **feature engineering** layer (changes, vol, z-scores, Sharpe, RSI, interactions)
- A binary **crisis vs normal** target derived from S&P 500 dynamics

This notebook is **Part I (intro)** of an 8-piece series.

### Series navigation

- **Part I:** Introduction  
- **Part II:** Data Preparation  
- **Part III:** Exploratory Data Analysis  
- **Part IV:** Feature Selection, Hyperparameter Tuning (LightGBM)  
- **Part V:** Model Evaluation and Interpretation (LightGBM)  
- **Part VI:** SVM and Neural Networks (MLP and 1D-CNN) — SVM \| MLP \| 1D-CNN  
- **Part VII:** Compare GBDT Models: XGBoost and LightGBM  
- **Part VIII:** Deployment of LightGBM Models (end-to-end process)  

---

## Predict Stock Market Crashes

This series is inspired by the following papers:

- Benhamou, Eric and Ohana, Jean-Jacques and Saltiel, David and Guez, Beatrice, *Planning in Financial Markets in Presence of Spikes: Using Machine Learning GBDT* (June 8, 2021). Université Paris-Dauphine Research Paper No. 3862428.  
  SSRN: https://ssrn.com/abstract=3862428 — DOI: http://dx.doi.org/10.2139/ssrn.3862428  
- Benhamou, Eric and Saltiel, David and Tabachnik, Serge and Bourdeix, Corentin and Chareyron, François and Guez, Beatrice, *Adaptive Supervised Learning for Volatility Targeting Models (Ecml Pkdd Midas 2021 Presentation Slides)* (September 18, 2021).  
  SSRN: https://ssrn.com/abstract=3926218 — DOI: http://dx.doi.org/10.2139/ssrn.3926218  
- Ungari, Sandrine and Benhamou, Eric, *Deep Reinforcement Learning for Portfolio Allocation* (July 14, 2021).  
  SSRN: https://ssrn.com/abstract=3886804  

---

## Define the target variable (crisis vs normal)

We assume two regimes:

- **Normal regime (0):** markets behave “normally”; long exposure is typically rewarded by equity risk premium.
- **Crisis regime (1):** stress/crash conditions; an asset manager should reduce exposure (or short, if allowed).

### Implementation used in this repo

The data prep script defines a crisis label using **S&P 500** (Yahoo ticker `^GSPC`):

1. Compute a 15-day percent change:
   $$
   r^{(15)}_t = \frac{P_t}{P_{t-15}} - 1
   $$

2. Compute a rolling 5th percentile threshold on a long window (approx. 2150 trading days), then flag “crisis” when the latest value is below that threshold:
   $$
   y_t = \mathbb{1}\Big[r^{(15)}_t < q_{0.05}\big(r^{(15)}_{t-2150:t-1}\big)\Big]
   $$

3. Create **forward-looking targets** (predicting future crisis from today’s features):
- `target_1d = regime_change.shift(-1)`
- `target_3d = regime_change.shift(-3)`
- `target_5d = regime_change.shift(-5)`
- `target_10d = regime_change.shift(-10)`

The main target used downstream is:
- **`target = target_3d`**

---

## Data sources (as implemented)

### Yahoo Finance (`yfinance`)
Downloaded from `start_date = 1996-01-01` to a **dynamic end date** (runtime `today()`).

Tickers used (can be edited in config):
- US equity + vol: `^GSPC`, `^DJI`, `^IXIC`, `^RUT`, `^NDX`, `^VIX`
- US yields: `^TYX`, `^TNX`, `^FVX`, `^IRX`
- FX proxies: `DX-Y.NYB`, `JPY=X`
- International equity: `EEM`, `IEMG`, `^FTSE`, `^STOXX50E`, `^N225`, `^KS11`, `^HSI`, `^KLSE`, `^N100`
- Commodities: `GC=F` (Gold), `CL=F` (WTI Oil)

**Cleaning rules:**
- Require `min_data_points = 100`
- Drop rows missing `Close`
- Use `auto_adjust=True`
- Remove timezone from indices

### FRED (`fredapi`)
FRED series are downloaded with `min_fred_points = 50` and aligned to the market calendar by:
- Forward-filling lower-frequency series (`W/M/Q/A`) onto daily dates
- Restricting to `start_date` .. `max_date` (max date of S&P target series)

FRED series list includes financial conditions, rates/spreads, macro, USD indices, and energy prices, e.g.:
`NFCI`, `STLFSI4`, `UMCSENT`, `TB3MS`, `T10Y2Y`, `CPFF`, `INDPRO`, `DTWEXBGS`, `DCOILWTICO`, etc.

> **Note:** the script expects a valid FRED API key (loaded via `fred_key.py`).

---

## Feature engineering (as implemented)

After merging all sources into a single daily table keyed by `x_date`, the script generates **per-column features** for each raw feature column (excluding targets and a few helper columns).

### Per-series engineered features

For each base column (price, yield, index level, macro series), we generate:

1. **Changes & percent changes** for lookbacks:
   - `lookback_periods = [5, 10, 20, 60, 120, 250]`
   - Features: `*_chg{p}`, `*_pct_chg{p}`
   - Extra indicator: for `p in {5, 20}`, add `*_above200dMA` (rolling “above mean” flag on the change series).

2. **Moving average distances / relative positioning**
   - `ma_dist20_50 = MA20 - MA50`
   - if enough history: `ma_dist200_50`, `price_vs_ma20`, `price_vs_ma200`

3. **Volatility proxies**
   - Rolling standard deviation: `*_std{p}` for `p in [60, 125]`
   - Log-return volatility (when series strictly positive): `*_volat{p}`

4. **Rolling Sharpe ratios**
   - `*_sharpe120`, `*_sharpe250` using annualization $$\sqrt{252}$$

5. **Rolling z-scores**
   - `zscore_periods = [60, 120, 200, 250]`
   - Feature: `*_zscore{p}`

6. **Momentum / RSI**
   - `*_roc20` (rate of change, percent)
   - `*_rsi` (14-day RSI)

If history is insufficient for a feature, the column is still created and set to `NaN` for **schema consistency**.

### Interaction features (focused set)

To avoid a combinatorial explosion, only a curated set of interactions is added:

- **Market vs VIX** ratios/products:
  - `{market}_div_VIX`, `{market}_mul_VIX`
- **S&P vs 10Y yield proxy**:
  - `GSPC_div_TNX` (only for S&P-related columns per script logic)
- **Commodities ratio**:
  - `Gold_Oil_Ratio = GC_F / CL_F`
- **Financial conditions interaction**:
  - `SP500_div_NFCI = GSPC / (NFCI + 2.0)` (offset to handle negative NFCI)
- **Dollar volume** (when volume exists):
  - `{ticker}_DollarVol = price * volume`

---

## Dataset assembly & cleaning

### Merge logic
1. Start from S&P-derived target table (includes `pct_chg15`, `regime_change`, `target_*`).
2. Join Yahoo tickers (Close mapped to ticker name; `Volume/High/Low` retained as `{ticker}_Volume`, etc.).
3. Merge FRED series on `x_date` after frequency normalization.

### Cleaning rules
- Drop rows with missing `target`
- Replace $$\pm\infty$$ with `NaN`
- Numeric columns: forward-fill, backward-fill, then fill remaining with `0`
- Drop columns with more than 50% missingness (before fill step where applicable)
- Deduplicate columns defensively

---

## Output artifact

The pipeline saves a single compressed parquet:

- **File:** `data.parquet`
- **Format:** PyArrow Parquet with `GZIP` compression
- **Storage optimization:** float64 columns cast to float32

This parquet is intended as the model-ready input for later parts (GBDT training, evaluation, deployment).

---

## Train/test split (series plan)

In later parts of the series we use **time-series aware** evaluation (expanding / walk-forward validation) rather than random splits to reduce leakage.

---

## Reproducibility notes

- Random seed is fixed: `RANDOM_SEED = 3407`
- End date is dynamic (today). For strict reproducibility, pin the end date in config.

---

## Disclaimer

This project is for research/education only and is **not investment advice**. Financial markets involve substantial risk.
```