---
layout: default
title: "Equity Market Crisis Regime Prediction"
parent: Crisis Prediction
grand_parent: Market Regime Analysis
has_children: false
has_toc: false
nav_order: 2
permalink: /docs/regime-analysis/crisis-prediction/crash-ml-models
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
