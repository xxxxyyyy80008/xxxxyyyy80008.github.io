---
layout: default
title: "MITSUI&CO. Commodity Prediction Challenge"
parent: Quant Competitions
nav_order: 2
has_children: false
permalink: /docs/quant-competitions/mitsui-commodity-prediction
---

# MITSUI&CO. Commodity Prediction Challenge (Kaggle)


- **Competition page:** [https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge)  
- [**Data:** ](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/data)  


Python Scripts for the **MITSUI&CO. Commodity Prediction Challenge**. [View](/notebooks/quant-competitions/20251003-commodity-prediction.html){:target="_blank" rel="noopener noreferrer"}

The implementation focuses on:

- Converting the wide, multi-asset dataset into a standardized **long format** (`date`, `asset_id`, OHLCV-like fields)
- Building simple rolling-window features per asset
- Training a multi-target model using **rank-transformed targets**
- Running inference with Kaggle’s provided inference server interface


## High-level approach

### 1) Data normalization (wide → long)
The raw `train.csv` contains many asset-specific columns. `DataProcessor.preprocess()` converts it into a single table with:

- `date` (renamed from `date_id`)
- `id` (string asset identifier, e.g., `FX_USDJPY`, `JPX_Gold_Mini`, `LME_AH`, `US_Stock_SPY`)
- Standardized fields (when available): `open`, `high`, `low`, `close`, `volume`, `sprice`, `interest`
- `id_num`: a numeric mapping of each asset ID for consistency

Assets covered in the mapping:
- **FX** pairs (39)
- **JPX** products (6)
- **LME** metals (4)
- **US stocks/ETFs** (a large list)

### 2) Feature engineering (rolling windows)
`DataProcessor.create_features()` computes per-asset features over windows defined by `Config.feature_windows = [5, 10, 20]`.

Feature families:
- **Pairwise normalized OHLC differences**:
  - For $$col1,col2 \in \{open,high,low,close\}$$ (with `col1 > col2`):
  - $$
  \frac{col1 - col2}{col1 + col2}
  $$
- **Shifted ratio**:
  - `open/close_shift1 = open / close.shift(1)`
- **Rolling return**:
  - `ret_w = close / close.shift(w) - 1`
- **Rolling volatility** (std of daily returns):
  - `vol_w = std(close/close.shift(1) - 1, window=w)`
- **Volume ratio**:
  - `volume_w = mean(volume,w) / mean(volume, 2w)`
- Two simple boolean “technical” signals (converted to floats) comparing today’s close/high/low to prior forward-filled bands:
  - `technical1_w`, `technical2_w`
- If present:
  - `sprice_change = sprice / sprice.shift(1) - 1`
  - `premium_discount = (close - sprice) / sprice`
  - `volume_interest_ratio = volume / (interest + 1)`

### 3) Model input layout (flatten by date)
`BaseModel.preprocess_features()` builds one feature vector per date by:
- grouping features by `date`
- taking each asset’s feature row and **flattening all assets’ features into a single 1D vector** for that date

This yields a matrix of shape roughly:
- `(#dates, #assets × #features_per_asset)`

Certain columns are excluded automatically:
- `date`, `id`, `label_d1`, `high`, `low`, `open`, `interest`, `sprice`, and any column containing `label`

### 4) Target transformation (rank-based)
Targets come from `train_labels.csv` with **424 target columns** (`NUM_TARGET_COLUMNS = 424`).

Before fitting, targets are converted into **rank-based values** per date:
- fill missing target values with 0
- compute ranks (double `argsort`) and normalize by 424

This matches the approach used in the script and is consistent with the competition’s rank-correlation-based evaluation.

### 5) Models included
The script defines three model classes:

- **`CatBoostModel`** (enabled by default)
  - `CatBoostRegressor(loss_function='MultiRMSE')`
  - Small depth and iteration count (baseline-style)
- **`LightGBMModelV2`** (defined, currently commented out in main)
  - Trains **one LightGBM model per target** (424 separate models)
- **`LightGBMModelV3`** (defined, currently commented out in main)
  - Similar to V2 with different hyperparameters

---
