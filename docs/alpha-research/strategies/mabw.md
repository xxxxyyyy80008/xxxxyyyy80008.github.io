---
layout: default
title:  MABW Volatility Breakout & Momentum Strategy
parent: Strategies & Backtests
nav_order: 1
---


### **MABW Volatility Breakout & Momentum Strategy**

This script implements, backtests, and analyzes a quantitative trading strategy designed to capitalize on volatility compression (squeezes) followed by momentum breakouts across a portfolio of equities.

### **1. Core Strategy Logic**
The strategy utilizes the **Moving Average Band Width (MABW)** to detect market phases.
*   **Indicators:** Calculates Fast (10) and Slow (60) Moving Averages to create bands. The "Width" is the percentage difference between the Upper and Lower bands.
*   **Entry Signal (Long Only):**
    *   **Momentum:** The 20-period EMA crosses *above* the Upper MABW Band.
    *   **Volatility Squeeze:** The `MAB_WIDTH` is at its lowest point over a specific lookback period (`MAB_LLV` - Lowest Low Value).
*   **Exit Signal:**
    *   **Volatility Expansion:** The `MAB_WIDTH` expands beyond a critical threshold (set to 30), indicating the move is over-extended or volatility has peaked.

### **2. Backtesting Engine**
The script contains a custom-built, event-driven backtesting engine with the following features:
*   **Data Source:** Fetches historical data via `yfinance` (Tickers: QQQ, JPM, AMD, JNJ, AAPL).
*   **Execution Simulation:** Uses **Next-Day Open** execution. Signals generated on Day $$T$$ result in trades executed at the Open of Day $$T+1$$.
*   **Capital Management:** Uses a "Fixed Capital per Ticker" approach. The initial capital ($100,000) is split equally among the tickers.
*   **Cost Modeling:** Simulates real-world friction with Commission (0.1%) and Slippage (0.05% + Fixed component).

### **3. Mathematical Components**
The MABW deviation is calculated using the root mean square of the difference between slow and fast moving averages:

$$
\text{Dev} = \sqrt{\text{Mean}((\text{MA}_{slow} - \text{MA}_{fast})^2)} \times \text{Multiplier}
$$

The Width is derived as:

$$
\text{Width} = \frac{\text{Upper Band} - \text{Lower Band}}{\text{MA}_{slow}} \times 100
$$

### **4. Reporting & Visualization**
The script generates extensive performance metrics and visualizations:
*   **Performance Metrics:** Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor.
*   **Visuals:** Equity curves, drawdown charts, monthly return heatmaps, and signal plots overlaying price data.
*   **Reporting:** Integrates `quantstats` to generate an HTML tear sheet comparing performance against a benchmark (SPY).
*   **Exports:** Saves trade logs, daily portfolio values, and metrics to CSV files.