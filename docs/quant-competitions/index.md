---
layout: default
title: "Quant Competitions"
nav_order: 5
has_children: true
has_toc: false
permalink: /docs/quant-competitions/
---

# Quant Competitions

A small collection of competition projects focused on **market microstructure**, **feature engineering**, and **robust evaluation / inference workflows**.

## Projects

### Optiver Realized Volatility Prediction (Kaggle)
- Links: [Competition](https://www.kaggle.com/c/optiver-realized-volatility-prediction) · [Data](https://www.kaggle.com/c/optiver-realized-volatility-prediction/data) · [Organizer tutorial](https://www.kaggle.com/jiashenliu/introduction-to-financial-concepts-and-data)
- Focus: realized-volatility modeling, correlation-based stock grouping, technical + `tsfresh` feature engineering, LightGBM training variants, and time-id-safe splitting to reduce leakage risk.
- Folder: `optiver-realized-volatility-prediction/`

### MITSUI&CO. Commodity Prediction Challenge (Kaggle)
- Focus: wide-to-long multi-asset normalization, rolling-window feature engineering, rank-transformed multi-target learning (424 targets), and a Kaggle inference-server compatible `predict()` pipeline.
- Folder: `mitsui-commodity-prediction-challenge/`

---
