---
title: custom auc fuction for xgboost and lightgbm models
---

### custom auc fuction for xgboost and lightgbm models

*Sep 10, 2020*

The script is copied from [Kaggle notebook here](https://github.com/jfpuget/Kaggle_Santander_2019/blob/master/notebooks/lgb_070_069_one_feature.ipynb
)

```python
#custom auc fuction for xgboost/lightgbm models
#copied from from: https://github.com/jfpuget/Kaggle_Santander_2019/blob/master/notebooks/lgb_070_069_one_feature.ipynb

import numpy as np 

def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc

def eval_auc (preds, dtrain):
    labels = dtrain.get_label()    
    return 'auc', fast_auc(labels, preds), True
```
