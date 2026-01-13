---
layout: default
title: "Optiver Realized Volatility Prediction"
parent: Quant Competitions
nav_order: 1
has_children: false
permalink: /docs/quant-competitions/optiver
---

# Optiver Realized Volatility Prediction (Kaggle)


- **Competition page:** [https://www.kaggle.com/c/optiver-realized-volatility-prediction](https://www.kaggle.com/c/optiver-realized-volatility-prediction)  
- **Data:** [https://www.kaggle.com/c/optiver-realized-volatility-prediction/data](https://www.kaggle.com/c/optiver-realized-volatility-prediction/data)  
- **Organizer tutorial (financial concepts + WAP):** [https://www.kaggle.com/jiashenliu/introduction-to-financial-concepts-and-data](https://www.kaggle.com/jiashenliu/introduction-to-financial-concepts-and-data)  

This folder contains my end-to-end work for the Optiver Realized Volatility Prediction competition: initial data exploration, correlation-based stock grouping, feature engineering (technical indicators + `tsfresh`), lightweight feature selection via correlation filtering, and LightGBM training setups under different stock subsets.

---

## Repository

[Source Code](https://github.com/xxxxyyyy80008/Kaggle-Optiver-Realized-Volatility-Prediction/tree/main/scripts){: .btn .btn-primary .fs-3 .mb-2 .mb-md-0}{:target="_blank" rel="noopener noreferrer"}

---


### 0) Initial exploration (data + target)
Goal: understand the dataset structure and what the model is predicting.

- `0_1_example.ipynb` — initial data exploration to understand the data and the problem setup  
- `0_2_target_analysis.ipynb` — target analysis (including target distribution)

---

### 1) Correlation analysis & stock grouping
Goal: use correlation structure to separate stocks into broad groups, then train models per group.

Initial grouping idea:
- **Group 1:** stocks with volatility highly correlated with each other  
- **Group 2:** stocks with volatility weakly correlated with each other  
- **Group 3:** stocks not clearly in the first two groups  

This grouping logic is later used to create “high-correlation” and “low-correlation/uncorrelated” stock lists for separate model training.

---

### 2) Feature engineering
My feature engineering approach focuses on:
- **Technical-indicator style features** (inspired by TA-Lib): https://github.com/mrjbq7/ta-lib  
- **Automated time-series features** (`tsfresh`): https://tsfresh.readthedocs.io/en/latest/

Files / notebooks:
- `technicals.py`  
  - Functions to create technical features  
  - Includes the **WAP** function (as used in the organizer tutorial)
- `1_1_prepare_book_data.ipynb` — prepares book data  
- `1_2_prepare_trade_data.ipynb` — prepares trade data  
- `2_1_calculate_corr_w_target.ipynb` — computes feature/target correlations  
- `2_2_filte_features_by_corr_w_target.ipynb` — filters features based on correlation to the target

---

## Feature selection (deadline-driven)
I joined the competition with only ~4–5 days remaining, so I did not use more sophisticated feature selection methods. Instead, I used **univariate Pearson correlation filtering**, which is fast but limited:
- captures **linear** relationships only  
- does **not** uncover non-linear relationships or feature interactions

I compute correlation of each feature against:
- the **raw target**
- **log(target)**
- a **normalized/outlier-removed** version of the target

High-level selection procedure:
1. Split stocks into **high-correlation** vs **low-correlation/uncorrelated** groups.
2. For each group:
   - take the top-*n* features (by correlation) per stock  
   - find **common** features across stocks (intersection of top-*n* lists)  
   - filter the resulting feature set to reduce **collinearity**
3. Combine feature lists from both groups.

---

## Training (LightGBM)
### Data prep
- `training/0_prep_data.ipynb`

### Model variants
- `training/lgb_10features_all.ipynb`  
  - all stocks  
  - target = **log(target)**  
  - outliers removed
- `training/lgb_10features_all2.ipynb`  
  - similar to `lgb_10features_all.ipynb`  
  - target = **raw target**
- `training/lgb_10features_highcorr.ipynb`  
  - stocks in **high-correlation** list  
  - target = **raw target**  
  - outliers removed
- `training/lgb_10features_highcorr2.ipynb`  
  - stocks in **high-correlation** list  
  - target = **log(target)**  
  - outliers removed
- `training/lgb_10features_lowcorr.ipynb`  
  - stocks in **low-correlation/uncorrelated** list  
  - target = **log(target)**  
  - outliers removed

### Predict / submission
- `script/3_predict_test.ipynb`  
  This notebook prepares the data, fits hyperparameters, trains models, and generates test predictions for submission.  
  I selected one tree based on results from `training/lgb_10features_highcorr.ipynb`. With more time, I would have run more systematic model selection and built a small ensemble (e.g., one model from each training setup).

---

## Train/test split policy (time-id safety)
The competition organizer notes: **“Time IDs are not necessarily sequential but are consistent across all stocks.”**

To avoid any potential information leakage (e.g., `time_id` appearing in train for one stock and test for another), I split **by randomly sampling `time_id`**:
- the overall train/test split is time-id based  
- K-fold splits within train are also time-id based  

This ensures each `time_id` across *all* stocks is assigned entirely to train or test.

---


