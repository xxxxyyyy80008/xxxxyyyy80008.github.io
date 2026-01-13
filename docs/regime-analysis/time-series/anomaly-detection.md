---
layout: default
title: "Anomaly Detection (Merlion): Isolation Forest, VAE, Ensemble"
parent: Market Regime Analysis
grand_parent: Time Series Modeling
nav_order: 3
permalink: /docs/regime-analysis/time-series/anomaly-detection
---

# Anomaly Detection with Salesforce Merlion (Unsupervised)

Applies **Salesforce Merlion** to generate anomaly scores on a market-derived time series using three unsupervised detectors:

- **Isolation Forest**
- **VAE** (Variational Autoencoder)
- **Ensemble** (Isolation Forest + VAE via `DetectorEnsemble`)

The goal is not labeled anomaly detection; instead, this analysis test whether anomaly scores carry information about *forward upside potential*.

## Data & target
- Data source: `yfinance`
- Asset: S&P 500 (`^GSPC`)
- Start date: `2000-01-01`
- Test split: last `400` observations

### Forward 20-day max return target
For each date $$t$$ with close $$C_t$$:

$$
target_t = 100 \times \frac{\max(C_{t+1}, \dots, C_{t+20}) - C_t}{C_t}
$$

Implementation detail: the max is computed over $$[t+1, t+20]$$ (starting tomorrow) to avoid look-ahead leakage.

---

## Method
### Models (Merlion)
- `IsolationForest(IsolationForestConfig())`
- `VAE(VAEConfig())`
- `DetectorEnsemble(...)` over both detectors, using `AggregateAlarms(alm_threshold=4)`

### Training & scoring
- Train on the target series as a univariate `TimeSeries` (no anomaly labels).
- Training anomaly scores are produced during `.train(...)`.
- Test anomaly scores are produced via `.get_anomaly_score(test_data)`.

### Evaluation (post-hoc)
This notebook treats the anomaly score as a candidate signal and reports:
- Correlation between $$target_t$$ and each anomaly score on **train**
- Correlation on **test**
- Train → test correlation drop as a rough generalization check

---

## Visual outputs
- Time series overlays (target vs anomaly scores) for train and test
- Correlation heatmaps (train/test)
- Score distribution comparisons (train/test)

---

## Key takeaways
- **VAE is the strongest single model in this run**: it shows the **highest correlation** with the forward $$20$$-day max return target on **both** the training and testing splits (relative to Isolation Forest and the ensemble).
- **Scale matters when comparing anomaly scores**: VAE produces **consistently larger-magnitude anomaly scores** than the other methods. This likely reflects **different score calibration/normalization** across detectors rather than “more anomalies” in an absolute sense.
- **Practical implication**: if you want a single unsupervised score that best tracks the forward-move proxy used here, **VAE is the best candidate** from these three. For downstream use, consider **standardizing scores** (e.g., z-score by a rolling window) before thresholding or combining with other signals.

---

## References
- Merlion: [https://github.com/salesforce/Merlion](https://github.com/salesforce/Merlion)  
- Merlion example: [https://github.com/salesforce/Merlion/blob/main/examples/anomaly/1_AnomalyFeatures.ipynb](https://github.com/salesforce/Merlion/blob/main/examples/anomaly/1_AnomalyFeatures.ipynb)  
- Isolation Forest (sklearn): [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)  
- Merlion VAE implementation: [https://github.com/salesforce/Merlion/blob/main/merlion/models/anomaly/vae.py](https://github.com/salesforce/Merlion/blob/main/merlion/models/anomaly/vae.py)  

---