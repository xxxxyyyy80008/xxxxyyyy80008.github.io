---
title: Quantative Finance Packages in Python - Aug 2022
---


## Quantative Finance Packages in Python - Aug 2022

*Aug 22, 2022*

reference: [awesome-quant](https://github.com/wilsonfreitas/awesome-quant)


### Data sources

packages that faciliates data scraping of various financial instruments

*As of Aug 22, 2022, these packages are still under active maintenance, i.e. last update was made within 12 months*


|  # |name   |description   |
| ------------ | ------------ | ------------ |
|  1 |  [yfinance](https://github.com/ranaroussi/yfinance) |  yfinance offers a threaded and Pythonic way to download market data from Yahoo!finance. |
|  **2**| **[akshare](https://github.com/akfamily/akshare)**  | an very efficient and easy to use library for a comprehensive list  of data sources (including stock prices of various markets such as CN and US) and functions. |
|  3 | [wallstreet](https://github.com/mcdallas/wallstreet)  |  Wallstreet is a Python 3 library for monitoring and analyzing real time Stock and Option data. Quotes are provided from the Google Finance API. |

### Visualization

|  # |name   |description   |
| ------------ | ------------ | ------------ |
| 1  |[dtale](https://github.com/man-group/dtale)   | D-Tale is the combination of a Flask back-end and a React front-end to bring you an easy way to view & analyze Pandas data structures. It integrates seamlessly with ipython notebooks & python/ipython terminals.   **It also faciliates network analysis!**|
| 2 | [mplfinance](https://github.com/matplotlib/mplfinance)|  based on matplotlib and it is tailored for visualization of financial data |
| 3  |  [finvizfinance](https://github.com/lit26/finvizfinance) | finvizfinance is a package which collects financial information from FinViz website. The package provides the information of the following: Stock charts, fundamental & technical information, insider information and stock news; Forex charts and performance; Crypto charts and performance  |

### Feature engineering

|  # |name   |description   |
| ------------ | ------------ | ------------ |
|  1 |[finta](https://github.com/peerchemist/finta)   |  lightweight library for technical indicators |
|  2 |  [ta-lib](https://github.com/mrjbq7/ta-lib)|  the core is build in C++ and wrapped by python. **finta** is becoming more comprehensive (in terms of technical indicators included) than ta-lib | 
|  3 |  [ta](https://github.com/bukosabino/ta) | almost all the technical indicators in this packages are also in finta  |
|4|[MyTT](https://github.com/mpquant/MyTT/blob/ea4f14857ecc46a3739a75ce2e6974b9057a6102/MyTT.py#L32)|it has some technical indicators not included in finta so far.|
|5|[tsfresh](https://github.com/blue-yonder/tsfresh)|The package provides systematic time-series feature extraction by combining established algorithms from statistics, time-series analysis, signal processing, and nonlinear dynamics with a robust feature selection algorithm. In this context, the term time-series is interpreted in the broadest possible sense, such that any types of sampled data or even event sequences can be characterised.|


### Backtesting and risk analysis

|  # |name   |description   |
| ------------ | ------------ | ------------ |
|  1 |[ffn](https://github.com/pmorissette/ffn)   |  this is a lightweight package with a short list of functions (such as calculation of information ratio, max drawdown) for backtesting.  |
|  2 |[Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib)   | Riskfolio-Lib is a library for making quantitative strategic asset allocation or portfolio optimization in Python   |
|  3 |  [Empyrial](https://github.com/ssantoshp/Empyrial) | similar to  *Riskfolio-Lib*  |
|4|[bt](https://github.com/pmorissette/bt)|bt is a flexible backtesting framework for Python used to test quantitative trading strategies. same developer as *ffn*|
|5|[Blankly](https://github.com/Blankly-Finance/Blankly)|Blankly is an ecosystem for algotraders enabling anyone to build, monetize and scale their trading algorithms for stocks, crypto, futures or forex. The same code can be backtested, paper traded, sandbox tested and run live by simply changing a single line.|
