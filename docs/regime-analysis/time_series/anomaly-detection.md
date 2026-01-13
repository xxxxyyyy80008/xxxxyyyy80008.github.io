---
layout: default
title: "Anomaly Detection (Merlion): Isolation Forest, VAE, Ensemble"
parent: Market Regime Analysis
grand_parent: Time Series Modeling
nav_order: 30
---

# Anomaly Detection with Salesforce Merlion (Unsupervised)
This project applies **Salesforce Merlion** to generate anomaly scores on a market-derived time series using three unsupervised detectors:

- **Isolation Forest**
- **VAE** (Variational Autoencoder)
- **Ensemble** (Isolation Forest + VAE via `DetectorEnsemble`)

The goal is not labeled anomaly detection; instead, we test whether anomaly scores carry information about *forward upside potential*.

## References
- Merlion: https://github.com/salesforce/Merlion  
- Merlion example: https://github.com/salesforce/Merlion/blob/main/examples/anomaly/1_AnomalyFeatures.ipynb  
- Isolation Forest (sklearn): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html  
- Merlion VAE implementation: https://github.com/salesforce/Merlion/blob/main/merlion/models/anomaly/vae.py  

---

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
- Time series overlays (target vs anomaly scores) for train and test (Plotly + Matplotlib)
- Correlation heatmaps (train/test)
- Score distribution comparisons (train/test)

If exported, an HTML artifact can be saved (example path from notes):
- `html/2_Merlion_Isoforest_VAE.html`

---

## Key takeaways
- **VAE** tends to show **higher correlation on training** but **much lower correlation on testing** versus Isolation Forest / ensemble.
- This pattern can indicate **overfitting** (strong in-sample fit, weaker out-of-sample generalization).
- Visual inspection often matches the correlation table: VAE looks “better” in train overlays but degrades on test.

---
