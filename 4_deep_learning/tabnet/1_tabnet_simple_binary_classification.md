---
sort: 2
title: TabNet - simple binary classification
---

### TabNet - simple binary classification

reference: 

https://github.com/dreamquark-ai/tabnet/blob/develop/census_example.ipynb

#### steps:
1. download market data using yfinance: download S&P 500 ('^GSPC')
1. calculate return 20-day max return (i.e. target in supervised learning problem):
   - for each date (T):
      - calculate the max price change in next 20 trading dates: price_change = (max{close price in T+1 to T+20} - {close price on T})/({close price on T})
1. convert the 20-day max return into binary target
1. engineer a few features
    - lag21: previous 21 day target
    - lag31: previous 31 day target
    - lag41: previous 41 day target
    - day price change: the difference between open and closing prices
        - (Close - Open)/Open
    - day max price change: the difference between high and low prices
        - (High-Low)/Open
    - one day close price change: day T close price versus day T-1 close price.
        - 100*({Close on T} - {Close on T-1})/{Close on T-1}
    - 10 day close price change: day T close price versus day T-10 close price.
        - 100*({Close on T} - {Close on T-10})/{Close on T-10}
    - 20 day close price change: day T close price versus day T-20 close price.
        - 100*({Close on T} - {Close on T-20})/{Close on T-20}
    - one day/10day/20day volume change

1. feed data into tabnet classifier
1. visualize the loss/performance in each epoch [html](html/tabnet_binary.html)


```python
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
import yfinance as yf #to download stock price data
```


```python
from pytorch_tabnet.tab_model import TabNetClassifier

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import os
from pathlib import Path
import shutil
```


```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```


```python
#initiate random seed
import random
def init_seed(random_seed):
    
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
init_seed(5678)
```

#### download S&P 500 price data


```python
ticker = '^GSPC'
cur_data = yf.Ticker(ticker)
hist = cur_data.history(period="max")
print(ticker, hist.shape, hist.index.min())
```

    ^GSPC (19721, 7) 1927-12-30 00:00:00
    


```python
df=hist[hist.index>='2000-01-01'].copy(deep=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividends</th>
      <th>Stock Splits</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-03</th>
      <td>1469.250000</td>
      <td>1478.000000</td>
      <td>1438.359985</td>
      <td>1455.219971</td>
      <td>931800000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>1455.219971</td>
      <td>1455.219971</td>
      <td>1397.430054</td>
      <td>1399.420044</td>
      <td>1009000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>1399.420044</td>
      <td>1413.270020</td>
      <td>1377.680054</td>
      <td>1402.109985</td>
      <td>1085500000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>1402.109985</td>
      <td>1411.900024</td>
      <td>1392.099976</td>
      <td>1403.449951</td>
      <td>1092300000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>1403.449951</td>
      <td>1441.469971</td>
      <td>1400.729980</td>
      <td>1441.469971</td>
      <td>1225200000</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### calcualte max return in next 20 trading days


```python
#for each stock_id, get the max close in next 20 trading days
price_col = 'Close'
roll_len=20
new_col = 'next_20day_max'
target_list = []

df.sort_index(ascending=True, inplace=True)
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividends</th>
      <th>Stock Splits</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-03</th>
      <td>1469.250000</td>
      <td>1478.000000</td>
      <td>1438.359985</td>
      <td>1455.219971</td>
      <td>931800000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>1455.219971</td>
      <td>1455.219971</td>
      <td>1397.430054</td>
      <td>1399.420044</td>
      <td>1009000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>1399.420044</td>
      <td>1413.270020</td>
      <td>1377.680054</td>
      <td>1402.109985</td>
      <td>1085500000</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_next20dmax=df[[price_col]].shift(1).rolling(roll_len).max()
df_next20dmax.columns=[new_col]
df = df.merge(df_next20dmax, right_index=True, left_index=True, how='inner')

df.dropna(how='any', inplace=True)
df['target']= 100*(df[new_col]-df[price_col])/df[price_col]  
```


```python
df['target'].describe()
```




    count    5479.000000
    mean        2.450868
    std         4.077580
    min        -3.743456
    25%         0.135604
    50%         1.130147
    75%         3.318523
    max        44.809803
    Name: target, dtype: float64




```python
df['target'].hist(bins=100)
```




    <AxesSubplot:>




    
![png](output_12_1.png)
    



```python
df['binary_target'] = 0
df.loc[df['target']>5, 'binary_target'] = 1
df['binary_target'].value_counts()
```




    0    4643
    1     836
    Name: binary_target, dtype: int64




```python
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividends</th>
      <th>Stock Splits</th>
      <th>next_20day_max</th>
      <th>target</th>
      <th>binary_target</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-02-01</th>
      <td>1394.459961</td>
      <td>1412.489990</td>
      <td>1384.790039</td>
      <td>1409.280029</td>
      <td>981000000</td>
      <td>0</td>
      <td>0</td>
      <td>1465.150024</td>
      <td>3.964435</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2000-02-02</th>
      <td>1409.280029</td>
      <td>1420.609985</td>
      <td>1403.489990</td>
      <td>1409.119995</td>
      <td>1038600000</td>
      <td>0</td>
      <td>0</td>
      <td>1465.150024</td>
      <td>3.976243</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2000-02-03</th>
      <td>1409.119995</td>
      <td>1425.780029</td>
      <td>1398.520020</td>
      <td>1424.969971</td>
      <td>1146500000</td>
      <td>0</td>
      <td>0</td>
      <td>1465.150024</td>
      <td>2.819712</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### create additional input features


```python
df['lag21']=df['target'].shift(21)
df['lag31']=df['target'].shift(31)
df['lag41']=df['target'].shift(41)

df['open_close_diff'] = df['Close'] - df['Open']
df['day_change']=(100*df['open_close_diff']/df['Open']).round(2)
df['day_max_change'] = (100*(df['High'] - df['Low'])/df['Open']).round(2)

#create a binary feature: 1 day change
#0: decrease; 1: increase
df['oneday_change']=(df['Close'].diff()>0)+1-1

df['10day_change']=df['Close'].diff(10)
df['20day_change']=df['Close'].diff(20)


df['oneday_volchange']=(df['Volume'].diff()>0)+1-1

df['10day_volchange']=df['Volume'].diff(10)
df['20day_volchange']=df['Volume'].diff(20)


df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividends</th>
      <th>Stock Splits</th>
      <th>next_20day_max</th>
      <th>target</th>
      <th>binary_target</th>
      <th>...</th>
      <th>lag41</th>
      <th>open_close_diff</th>
      <th>day_change</th>
      <th>day_max_change</th>
      <th>oneday_change</th>
      <th>10day_change</th>
      <th>20day_change</th>
      <th>oneday_volchange</th>
      <th>10day_volchange</th>
      <th>20day_volchange</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-02-01</th>
      <td>1394.459961</td>
      <td>1412.489990</td>
      <td>1384.790039</td>
      <td>1409.280029</td>
      <td>981000000</td>
      <td>0</td>
      <td>0</td>
      <td>1465.150024</td>
      <td>3.964435</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>14.820068</td>
      <td>1.06</td>
      <td>1.99</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-02-02</th>
      <td>1409.280029</td>
      <td>1420.609985</td>
      <td>1403.489990</td>
      <td>1409.119995</td>
      <td>1038600000</td>
      <td>0</td>
      <td>0</td>
      <td>1465.150024</td>
      <td>3.976243</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.160034</td>
      <td>-0.01</td>
      <td>1.21</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-02-03</th>
      <td>1409.119995</td>
      <td>1425.780029</td>
      <td>1398.520020</td>
      <td>1424.969971</td>
      <td>1146500000</td>
      <td>0</td>
      <td>0</td>
      <td>1465.150024</td>
      <td>2.819712</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>15.849976</td>
      <td>1.12</td>
      <td>1.93</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 22 columns</p>
</div>




```python
df['day_change'].hist(bins=50)
```




    <AxesSubplot:>




    
![png](output_17_1.png)
    



```python
#convert day_change into categorical feature
#above 2- class 1; below -2 - class -1, in the middle - class0
df['day_change_cat']=0
df.loc[df['day_change']<=-2, 'day_change_cat']=-1
df.loc[df['day_change']>=2, 'day_change_cat']=1
df['day_change_cat'].value_counts()
```




     0    5095
    -1     210
     1     174
    Name: day_change_cat, dtype: int64




```python
df.dropna(how='any', inplace=True)
print(df.shape, df.index.min())
df.head(3)
```

    (5438, 23) 2000-03-30 00:00:00
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividends</th>
      <th>Stock Splits</th>
      <th>next_20day_max</th>
      <th>target</th>
      <th>binary_target</th>
      <th>...</th>
      <th>open_close_diff</th>
      <th>day_change</th>
      <th>day_max_change</th>
      <th>oneday_change</th>
      <th>10day_change</th>
      <th>20day_change</th>
      <th>oneday_volchange</th>
      <th>10day_volchange</th>
      <th>20day_volchange</th>
      <th>day_change_cat</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-03-30</th>
      <td>1508.520020</td>
      <td>1517.380005</td>
      <td>1474.630005</td>
      <td>1487.920044</td>
      <td>1193400000</td>
      <td>0</td>
      <td>0</td>
      <td>1527.459961</td>
      <td>2.657395</td>
      <td>0</td>
      <td>...</td>
      <td>-20.599976</td>
      <td>-1.37</td>
      <td>2.83</td>
      <td>0</td>
      <td>29.450073</td>
      <td>106.160034</td>
      <td>1</td>
      <td>-288900000.0</td>
      <td>-5200000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2000-03-31</th>
      <td>1487.920044</td>
      <td>1519.810059</td>
      <td>1484.380005</td>
      <td>1498.579956</td>
      <td>1227400000</td>
      <td>0</td>
      <td>0</td>
      <td>1527.459961</td>
      <td>1.927158</td>
      <td>0</td>
      <td>...</td>
      <td>10.659912</td>
      <td>0.72</td>
      <td>2.38</td>
      <td>1</td>
      <td>34.109985</td>
      <td>89.409912</td>
      <td>1</td>
      <td>-67700000.0</td>
      <td>77100000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2000-04-03</th>
      <td>1498.579956</td>
      <td>1507.189941</td>
      <td>1486.959961</td>
      <td>1505.969971</td>
      <td>1021700000</td>
      <td>0</td>
      <td>0</td>
      <td>1527.459961</td>
      <td>1.426987</td>
      <td>0</td>
      <td>...</td>
      <td>7.390015</td>
      <td>0.49</td>
      <td>1.35</td>
      <td>1</td>
      <td>49.339966</td>
      <td>114.689941</td>
      <td>0</td>
      <td>100900000.0</td>
      <td>-7300000.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 23 columns</p>
</div>



# split data into simple training and testing subsets


```python
target='binary_target'
bool_columns = ['oneday_change', 'oneday_volchange']
```


```python
df.dropna(how='any', inplace=True)
train = df.copy(deep=True)
```

# Simple preprocessing

Label encode categorical features and fill empty cells.


```python
categorical_columns = ['day_change_cat']
categorical_dims =  {}
for col in categorical_columns:
    print(col, train[col].nunique())
    l_enc = LabelEncoder()
    train[col] = l_enc.fit_transform(train[col].values)
    
    categorical_dims[col] = len(l_enc.classes_)

categorical_dims
```

    day_change_cat 3
    




    {'day_change_cat': 3}




```python
categorical_columns, categorical_dims
```




    (['day_change_cat'], {'day_change_cat': 3})



# Define categorical features for categorical embeddings


```python
unused_feat = ['Dividends', 'Stock Splits', 'next_20day_max',
               'open_close_diff', 'day_change' ]

features = [ col for col in train.columns if col not in unused_feat+[target]] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

```


```python
print(features)
```

    ['Open', 'High', 'Low', 'Close', 'Volume', 'target', 'lag21', 'lag31', 'lag41', 'day_max_change', 'oneday_change', '10day_change', '20day_change', 'oneday_volchange', '10day_volchange', '20day_volchange', 'day_change_cat']
    


```python
cat_idxs
```




    [16]




```python
cat_dims
```




    [3]



# Network parameters


```python
clf = TabNetClassifier(
    n_d=32, n_a=32, n_steps=5,
    gamma=1.5, n_independent=2, n_shared=2,
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=1,
    lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params = {"gamma": 0.95,
                     "step_size": 20},
    scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15
)
```

    Device used : cpu
    


```python
clf2 = TabNetClassifier()
clf2
```

    Device used : cpu
    




    TabNetClassifier(n_d=8, n_a=8, n_steps=3, gamma=1.3, cat_idxs=[], cat_dims=[], cat_emb_dim=1, n_independent=2, n_shared=2, epsilon=1e-15, momentum=0.02, lambda_sparse=0.001, seed=0, clip_value=1, verbose=1, optimizer_fn=<class 'torch.optim.adam.Adam'>, optimizer_params={'lr': 0.02}, scheduler_fn=None, scheduler_params={}, mask_type='sparsemax', input_dim=None, output_dim=None, device_name='auto')



# Training


```python
train.shape
```




    (5438, 23)




```python
X_train = train[features].values[:-1500,:]
y_train = train[target].values[:-1500]

X_valid = train[features].values[-1450:-650,:]
y_valid = train[target].values[-1450:-650]

X_test = train[features].values[-600:, ]
y_test = train[target].values[-600:]
```


```python
X_train.shape, X_valid.shape, X_test.shape
```




    ((3938, 17), (800, 17), (600, 17))




```python
max_epochs = 50
```


```python
clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    max_epochs=max_epochs, patience=100,
    batch_size=1024, virtual_batch_size=256
) 
```

    epoch 0  | loss: 0.6085  | train_auc: 0.5946  | valid_auc: 0.55134 |  0:00:00s
    epoch 1  | loss: 0.25079 | train_auc: 0.72439 | valid_auc: 0.62383 |  0:00:01s
    epoch 2  | loss: 0.20278 | train_auc: 0.74185 | valid_auc: 0.67725 |  0:00:02s
    epoch 3  | loss: 0.16259 | train_auc: 0.77868 | valid_auc: 0.73814 |  0:00:03s
    epoch 4  | loss: 0.1455  | train_auc: 0.823   | valid_auc: 0.88184 |  0:00:03s
    epoch 5  | loss: 0.12248 | train_auc: 0.86481 | valid_auc: 0.80974 |  0:00:04s
    epoch 6  | loss: 0.12148 | train_auc: 0.87193 | valid_auc: 0.73214 |  0:00:05s
    epoch 7  | loss: 0.09959 | train_auc: 0.92461 | valid_auc: 0.855   |  0:00:05s
    epoch 8  | loss: 0.1053  | train_auc: 0.90809 | valid_auc: 0.90169 |  0:00:06s
    epoch 9  | loss: 0.10171 | train_auc: 0.92914 | valid_auc: 0.84598 |  0:00:07s
    epoch 10 | loss: 0.09968 | train_auc: 0.88691 | valid_auc: 0.92718 |  0:00:07s
    epoch 11 | loss: 0.07191 | train_auc: 0.88264 | valid_auc: 0.90451 |  0:00:08s
    epoch 12 | loss: 0.07202 | train_auc: 0.89332 | valid_auc: 0.91838 |  0:00:09s
    epoch 13 | loss: 0.06804 | train_auc: 0.9532  | valid_auc: 0.9241  |  0:00:10s
    epoch 14 | loss: 0.05959 | train_auc: 0.97328 | valid_auc: 0.9259  |  0:00:10s
    epoch 15 | loss: 0.05237 | train_auc: 0.9798  | valid_auc: 0.92061 |  0:00:11s
    epoch 16 | loss: 0.0527  | train_auc: 0.98212 | valid_auc: 0.95056 |  0:00:12s
    epoch 17 | loss: 0.04247 | train_auc: 0.98444 | valid_auc: 0.96823 |  0:00:12s
    epoch 18 | loss: 0.04855 | train_auc: 0.98537 | valid_auc: 0.91772 |  0:00:13s
    epoch 19 | loss: 0.05294 | train_auc: 0.98751 | valid_auc: 0.97337 |  0:00:14s
    epoch 20 | loss: 0.05513 | train_auc: 0.98903 | valid_auc: 0.95136 |  0:00:15s
    epoch 21 | loss: 0.05105 | train_auc: 0.98586 | valid_auc: 0.95794 |  0:00:15s
    epoch 22 | loss: 0.05698 | train_auc: 0.98526 | valid_auc: 0.95633 |  0:00:16s
    epoch 23 | loss: 0.04063 | train_auc: 0.98628 | valid_auc: 0.94675 |  0:00:17s
    epoch 24 | loss: 0.04106 | train_auc: 0.99017 | valid_auc: 0.96589 |  0:00:17s
    epoch 25 | loss: 0.04789 | train_auc: 0.99188 | valid_auc: 0.96874 |  0:00:18s
    epoch 26 | loss: 0.03689 | train_auc: 0.99214 | valid_auc: 0.95453 |  0:00:19s
    epoch 27 | loss: 0.04332 | train_auc: 0.99126 | valid_auc: 0.96753 |  0:00:20s
    epoch 28 | loss: 0.04386 | train_auc: 0.9947  | valid_auc: 0.98204 |  0:00:20s
    epoch 29 | loss: 0.03909 | train_auc: 0.99469 | valid_auc: 0.99208 |  0:00:21s
    epoch 30 | loss: 0.04479 | train_auc: 0.99478 | valid_auc: 0.9625  |  0:00:22s
    epoch 31 | loss: 0.04565 | train_auc: 0.99615 | valid_auc: 0.9235  |  0:00:22s
    epoch 32 | loss: 0.03824 | train_auc: 0.99425 | valid_auc: 0.98025 |  0:00:23s
    epoch 33 | loss: 0.02534 | train_auc: 0.99519 | valid_auc: 0.99611 |  0:00:24s
    epoch 34 | loss: 0.02821 | train_auc: 0.99603 | valid_auc: 0.99306 |  0:00:25s
    epoch 35 | loss: 0.03126 | train_auc: 0.99478 | valid_auc: 0.99035 |  0:00:25s
    epoch 36 | loss: 0.02935 | train_auc: 0.9962  | valid_auc: 0.9991  |  0:00:26s
    epoch 37 | loss: 0.04743 | train_auc: 0.99714 | valid_auc: 0.99868 |  0:00:27s
    epoch 38 | loss: 0.04275 | train_auc: 0.99752 | valid_auc: 0.99935 |  0:00:27s
    epoch 39 | loss: 0.03126 | train_auc: 0.99728 | valid_auc: 0.99664 |  0:00:28s
    epoch 40 | loss: 0.03763 | train_auc: 0.99903 | valid_auc: 0.99352 |  0:00:29s
    epoch 41 | loss: 0.05079 | train_auc: 0.999   | valid_auc: 0.9938  |  0:00:29s
    epoch 42 | loss: 0.03019 | train_auc: 0.99848 | valid_auc: 0.99407 |  0:00:30s
    epoch 43 | loss: 0.02432 | train_auc: 0.99936 | valid_auc: 0.98669 |  0:00:31s
    epoch 44 | loss: 0.04208 | train_auc: 0.99914 | valid_auc: 0.98569 |  0:00:32s
    epoch 45 | loss: 0.03639 | train_auc: 0.99944 | valid_auc: 0.98695 |  0:00:32s
    epoch 46 | loss: 0.03375 | train_auc: 0.99922 | valid_auc: 0.9797  |  0:00:33s
    epoch 47 | loss: 0.03105 | train_auc: 0.99863 | valid_auc: 0.98265 |  0:00:34s
    epoch 48 | loss: 0.02795 | train_auc: 0.99946 | valid_auc: 0.98412 |  0:00:34s
    epoch 49 | loss: 0.03275 | train_auc: 0.9993  | valid_auc: 0.98    |  0:00:35s
    Stop training because you reached max_epochs = 50 with best_epoch = 38 and best_valid_auc = 0.99935
    Best weights from best epoch are automatically used!
    


```python
clf2.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    max_epochs=max_epochs, patience=50,
    batch_size=1024, virtual_batch_size=128
) 
```

    epoch 0  | loss: 0.04553 | train_auc: 0.98935 | valid_auc: 0.99878 |  0:00:00s
    epoch 1  | loss: 0.04278 | train_auc: 0.98504 | valid_auc: 0.97955 |  0:00:00s
    epoch 2  | loss: 0.02881 | train_auc: 0.98514 | valid_auc: 0.99199 |  0:00:00s
    epoch 3  | loss: 0.04076 | train_auc: 0.98423 | valid_auc: 0.99251 |  0:00:01s
    epoch 4  | loss: 0.03414 | train_auc: 0.98811 | valid_auc: 0.99318 |  0:00:01s
    epoch 5  | loss: 0.03644 | train_auc: 0.99269 | valid_auc: 0.99511 |  0:00:01s
    epoch 6  | loss: 0.02725 | train_auc: 0.9956  | valid_auc: 0.99439 |  0:00:01s
    epoch 7  | loss: 0.0272  | train_auc: 0.99641 | valid_auc: 0.99542 |  0:00:02s
    epoch 8  | loss: 0.03312 | train_auc: 0.99755 | valid_auc: 0.99533 |  0:00:02s
    epoch 9  | loss: 0.03144 | train_auc: 0.9983  | valid_auc: 0.99587 |  0:00:02s
    epoch 10 | loss: 0.03173 | train_auc: 0.99891 | valid_auc: 0.9969  |  0:00:02s
    epoch 11 | loss: 0.02985 | train_auc: 0.99912 | valid_auc: 0.99827 |  0:00:03s
    epoch 12 | loss: 0.02482 | train_auc: 0.99905 | valid_auc: 0.99755 |  0:00:03s
    epoch 13 | loss: 0.02069 | train_auc: 0.99904 | valid_auc: 0.99718 |  0:00:03s
    epoch 14 | loss: 0.02969 | train_auc: 0.99907 | valid_auc: 0.99257 |  0:00:03s
    epoch 15 | loss: 0.02153 | train_auc: 0.999   | valid_auc: 0.98468 |  0:00:04s
    epoch 16 | loss: 0.01556 | train_auc: 0.99883 | valid_auc: 0.97252 |  0:00:04s
    epoch 17 | loss: 0.02614 | train_auc: 0.99903 | valid_auc: 0.94694 |  0:00:04s
    epoch 18 | loss: 0.02147 | train_auc: 0.99924 | valid_auc: 0.9476  |  0:00:05s
    epoch 19 | loss: 0.03949 | train_auc: 0.99965 | valid_auc: 0.97237 |  0:00:05s
    epoch 20 | loss: 0.03701 | train_auc: 0.99963 | valid_auc: 0.9469  |  0:00:05s
    epoch 21 | loss: 0.05568 | train_auc: 0.99977 | valid_auc: 0.93323 |  0:00:06s
    epoch 22 | loss: 0.02871 | train_auc: 0.99986 | valid_auc: 0.94559 |  0:00:06s
    epoch 23 | loss: 0.02558 | train_auc: 0.99988 | valid_auc: 1.0     |  0:00:06s
    epoch 24 | loss: 0.01854 | train_auc: 0.9999  | valid_auc: 1.0     |  0:00:06s
    epoch 25 | loss: 0.04108 | train_auc: 0.99992 | valid_auc: 1.0     |  0:00:07s
    epoch 26 | loss: 0.02608 | train_auc: 0.99991 | valid_auc: 1.0     |  0:00:07s
    epoch 27 | loss: 0.0196  | train_auc: 0.99992 | valid_auc: 1.0     |  0:00:07s
    epoch 28 | loss: 0.01689 | train_auc: 0.99994 | valid_auc: 1.0     |  0:00:07s
    epoch 29 | loss: 0.02221 | train_auc: 0.99993 | valid_auc: 1.0     |  0:00:08s
    epoch 30 | loss: 0.01336 | train_auc: 0.99988 | valid_auc: 1.0     |  0:00:08s
    epoch 31 | loss: 0.01686 | train_auc: 0.99994 | valid_auc: 1.0     |  0:00:08s
    epoch 32 | loss: 0.01983 | train_auc: 0.99994 | valid_auc: 1.0     |  0:00:09s
    epoch 33 | loss: 0.01631 | train_auc: 0.99997 | valid_auc: 0.99957 |  0:00:09s
    epoch 34 | loss: 0.02721 | train_auc: 0.99994 | valid_auc: 0.99994 |  0:00:09s
    epoch 35 | loss: 0.02679 | train_auc: 0.99985 | valid_auc: 1.0     |  0:00:09s
    epoch 36 | loss: 0.04    | train_auc: 0.99984 | valid_auc: 1.0     |  0:00:10s
    epoch 37 | loss: 0.03121 | train_auc: 0.99996 | valid_auc: 0.99984 |  0:00:10s
    epoch 38 | loss: 0.03083 | train_auc: 0.99997 | valid_auc: 0.99904 |  0:00:10s
    epoch 39 | loss: 0.02723 | train_auc: 0.99994 | valid_auc: 0.99827 |  0:00:10s
    epoch 40 | loss: 0.0429  | train_auc: 0.99991 | valid_auc: 0.99788 |  0:00:11s
    epoch 41 | loss: 0.02929 | train_auc: 0.99994 | valid_auc: 0.99902 |  0:00:11s
    epoch 42 | loss: 0.03338 | train_auc: 0.99997 | valid_auc: 1.0     |  0:00:11s
    epoch 43 | loss: 0.03398 | train_auc: 0.99998 | valid_auc: 1.0     |  0:00:11s
    epoch 44 | loss: 0.0267  | train_auc: 0.99995 | valid_auc: 0.9998  |  0:00:12s
    epoch 45 | loss: 0.02937 | train_auc: 0.99991 | valid_auc: 0.99996 |  0:00:12s
    epoch 46 | loss: 0.02838 | train_auc: 0.99996 | valid_auc: 1.0     |  0:00:12s
    epoch 47 | loss: 0.01855 | train_auc: 0.99998 | valid_auc: 1.0     |  0:00:13s
    epoch 48 | loss: 0.02415 | train_auc: 0.99998 | valid_auc: 1.0     |  0:00:13s
    epoch 49 | loss: 0.01826 | train_auc: 0.99998 | valid_auc: 1.0     |  0:00:13s
    Stop training because you reached max_epochs = 50 with best_epoch = 23 and best_valid_auc = 1.0
    Best weights from best epoch are automatically used!
    


```python
fig_list =[]
```


```python
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])
           
x_vals=list(range(1, max_epochs+1))

fig.add_trace(go.Scatter(
                        name="loss",
                        mode="lines", x=x_vals, y=clf.history['loss']),
              secondary_y=False
             )

fig.add_trace(go.Scatter(
                        name="train_auc",
                        mode="lines", x=x_vals,y=clf.history['train_auc']),
              secondary_y=True
             )

fig.add_trace(go.Scatter(
                        name="valid_auc",
                        mode="lines", x=x_vals,y=clf.history['valid_auc']),
              secondary_y=True
             )


fig.update_layout(hovermode="x unified", 
                  title_text="training data - loss and auc"
                 )



#fig.show()

fig_list.append(fig)
```


```python
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])
           
x_vals=list(range(1, max_epochs+1))

fig.add_trace(go.Scatter(
                        name="loss",
                        mode="lines", x=x_vals, y=clf2.history['loss']),
              secondary_y=False
             )

fig.add_trace(go.Scatter(
                        name="train_auc",
                        mode="lines", x=x_vals,y=clf2.history['train_auc']),
              secondary_y=True
             )

fig.add_trace(go.Scatter(
                        name="valid_auc",
                        mode="lines", x=x_vals,y=clf2.history['valid_auc']),
              secondary_y=True
             )


fig.update_layout(hovermode="x unified", 
                  title_text="training data - loss and auc - default hyperparameters"
                 )



#fig.show()

fig_list.append(fig)
```

### Predictions



```python

preds_mapper = { idx : class_name for idx, class_name in enumerate(clf.classes_)}
preds = clf.predict_proba(X_test)
y_pred = np.vectorize(preds_mapper.get)(np.argmax(preds, axis=1))
test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)

preds_mapper2 = { idx : class_name for idx, class_name in enumerate(clf2.classes_)}
preds2 = clf2.predict_proba(X_test)
y_pred2 = np.vectorize(preds_mapper2.get)(np.argmax(preds2, axis=1))
test_acc2 = accuracy_score(y_pred=y_pred2, y_true=y_test)


print(f"BEST VALID SCORE FOR  : {clf.best_cost}, {clf2.best_cost}")
print(f"FINAL TEST SCORE FOR  : {test_acc}, {test_acc2}")
```

    BEST VALID SCORE FOR  : 0.9993484148154181, 1.0
    FINAL TEST SCORE FOR  : 0.9733333333333334, 0.9483333333333334
    


```python
y_pred = clf.predict(X_test)
test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)


y_pred2 = clf2.predict(X_test)
test_acc2 = accuracy_score(y_pred=y_pred2, y_true=y_test)
print(f"FINAL TEST SCORE FOR  : {test_acc}, {test_acc2}")
```

    FINAL TEST SCORE FOR  : 0.9733333333333334, 0.9483333333333334
    

# Save and load Model


```python
# save state dict
saved_filename = clf.save_model('binary_model')
```

    Successfully saved model at binary_model.zip
    


```python
# define new model and load save parameters
loaded_clf = TabNetClassifier()
loaded_clf.load_model(saved_filename)
```

    Device used : cpu
    Device used : cpu
    


```python
loaded_preds = loaded_clf.predict_proba(X_test)
loaded_y_pred = np.vectorize(preds_mapper.get)(np.argmax(loaded_preds, axis=1))

loaded_test_acc = accuracy_score(y_pred=loaded_y_pred, y_true=y_test)

print(f"FINAL TEST SCORE FOR  : {loaded_test_acc}")
```

    FINAL TEST SCORE FOR  : 0.9733333333333334
    


```python
test_acc == loaded_test_acc
```




    True



# Global explainability : feat importance summing to 1


```python
clf.feature_importances_
```




    array([0.01141833, 0.10568571, 0.01697402, 0.0599244 , 0.0296432 ,
           0.2647944 , 0.01209274, 0.03980985, 0.02885467, 0.00494661,
           0.04379065, 0.04271051, 0.20845756, 0.00243536, 0.0882974 ,
           0.02919195, 0.01097264])



# Local explainability and masks


```python
from matplotlib import pyplot as plt
```


```python
explain_matrix, masks = clf.explain(X_test)
```


```python
fig, axs = plt.subplots(1, 5, figsize=(20,20))

for i in range(5):
    axs[i].imshow(masks[i][:50])
    axs[i].set_title(f"mask {i}")
```


    
![png](output_57_0.png)
    


#### Export graphs to a html file


```python
fig_path = r'html/tabnet_binary.html'
fig_list[0].write_html(fig_path)


with open(fig_path, 'a') as f:
    for fig_i in fig_list[1:]:
        f.write(fig_i.to_html(full_html=False, include_plotlyjs='cdn'))
```
