---
title: Predicting regime shift in time series data - Literature review
---


***
### Predicting regime shift in time series data - Literature review
*Nov 6, 2021*


#### Literature review

- Rodionov, S. N.: A brief overview of the regime shift detection methods
	- [PDF](https://www.beringclimate.noaa.gov/regimes/rodionov_overview.pdf)
	- this article is a nice summary of statistical methods to detect regime shifts: “The methods reviewed here are primarily those that are used in atmospheric and oceanic (physical and biological) studies”
	- 4 types of shifts: shifts in mean, variance, frequency structure and system
	- for each type of shift, there are various statistical tests or methods, each has its pros, cons, and limitations
	- for example, for shifts in mean, there are t-test, Bayesian analysis, man-whitney u-test, and so on. 
- Markov switching autoregressive models/Markov switching dynamic regression models
	- [statmodels](https://www.statsmodels.org/dev/examples/notebooks/generated/markov_regression.html)
	- [stata](https://www.stata.com/manuals14/tsmswitch.pdf)
	- “Markov-switching models are used for series that are believed to transition over a finite set of unobserved states, allowing the process to evolve differently in each state. The transitions occur according to a Markov process. The time of transition from one state to another and the duration between changes in state is random. For example, these models can be used to understand the process that governs the time at which economic growth transitions between expansion and recession and the duration of each period.”
	- “Markov-switching models are widely applied in the social sciences. For example, in economics, the growth rate of Gross Domestic Product is modeled as a switching process to capture the asymmetrical behavior observed over expansions and recessions. Other economic examples include modeling interest rates and exchange rates.”
- Anomaly detection/outlier detection by machine learning (deep learning) models
	- anomaly detection resources:
	   - this is a very nice repository of academic papers, python packages and datasets on anomaly detection, including time series data anomaly detection
	   - [github](https://github.com/yzhao062/anomaly-detection-resources)
	- Merlion: A Machine Learning Library for Time Series
	   - [Paper](https://arxiv.org/pdf/2109.09265.pdf)
	   - [github](https://github.com/salesforce/Merlion)
	   - This is a fairly new (as of Nov 2021) machine learning library for time series developed by Salesforce.
	   - ![Architecture of modules in Merlion](../img/merlion.png)
- Kaggle
	- Weather forecasting winning solution [link](https://github.com/fengyang95/AIC_Weather_Forecasting)
	- Kaggle Web Traffic Time Series Forecasting 1st place solution [link](https://github.com/Arturus/kaggle-web-traffic/blob/master/model.py)
