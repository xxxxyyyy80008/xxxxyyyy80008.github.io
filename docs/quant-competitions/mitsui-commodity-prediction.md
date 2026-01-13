---
layout: default
title: "MITSUI&CO. Commodity Prediction Challenge"
parent: Quant Competitions
nav_order: 2
has_children: false
---

# MITSUI&CO. Commodity Prediction Challenge (Kaggle)


- **Competition page:** https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge  
- **Data:** https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/data  


Python Scripts for the **MITSUI&CO. Commodity Prediction Challenge**. The implementation focuses on:

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

## Evaluation metric (as implemented here)

The code includes a local scoring implementation:
- Computes **daily rank correlation** between predictions and true targets across the 424 targets
- Returns a **Sharpe ratio** of daily rank correlations:

$$
Sharpe = \frac{\mathbb{E}[\rho_t]}{\sqrt{\mathrm{Var}(\rho_t)}}
$$

Handling missing targets:
- The competition uses a sentinel `SOLUTION_NULL_FILLER = -999999` in the solution; this is converted back to nulls before computing correlations.

---

## Inference / submission logic

### Streaming-style prediction function
The `predict(...)` function is written for Kaggle’s inference server interface. It:

1. Receives a `test` batch for a single `date_id`
2. Appends this batch to a global rolling store `df_train_original` (keeping only prior dates)
3. Builds features on a rolling history:
   - `N_last_day = 60`
   - uses `.tail(N_last_day * 143)` before preprocessing
   - uses `.tail(143 * 10)` after feature creation
4. Produces predictions for `target_0 ... target_423` for the latest date

Currently, the ensemble is effectively **CatBoost-only**:
- `ensemble_prediction = prediction1`  
(The LightGBM models are present but commented out.)

### Local inference server
The script attempts to run:
- `kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)`
- either `serve()` during a competition rerun or `run_local_gateway(...)` otherwise

---

## How to run

### 1) Install dependencies
This script uses:
- `numpy`, `pandas`, `polars`
- `catboost`
- optionally `lightgbm`
- Kaggle evaluation server (`kaggle_evaluation`) when running the local gateway

Example:
```bash
pip install numpy pandas polars catboost lightgbm
```

### 2) Dataset path
The code resolves input from either:
- local folder: `./mitsui-commodity-prediction-challenge/`
- or Kaggle input: `/kaggle/input/mitsui-commodity-prediction-challenge/`

Ensure `train.csv` and `train_labels.csv` exist under that directory.

### 3) Train
Training is invoked by:
- loading and preprocessing `train.csv`
- creating features
- calling `model1.train(df_features, config.num_valid)`

Validation size:
- `config.num_valid = 134` (last 134 dates used as validation in this baseline)

---

## Notes / current limitations

- **Validation**: the script scores only the first 90 rows of the last validation segment in one place (`head(90)`), consistent with the provided code.
- **Ensembling**: LightGBM variants exist but are commented out in training and inference.
- **Feature missingness**: many assets do not have full OHLCV fields; the preprocessing fills unavailable fields with NaN and features may carry NaNs depending on asset type.
- **Performance**: LightGBM v2/v3 train 424 models; this is significantly heavier than the CatBoost baseline.
