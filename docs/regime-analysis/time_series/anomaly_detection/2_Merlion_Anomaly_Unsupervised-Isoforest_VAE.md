---
layout: default
title: Anomaly Detection with Salesforce Merlion Package - Unsupervised learning with Isolation Forest, VAE, and Ensemble
parent: Time Series Anomaly Detection
grand_parent: Time Series
nav_order: 3
---


# Anomaly Detection with Salesforce Merlion Package - Unsupervised learning with Isolation Forest, VAE, and Ensemble

### Reference:
- github: https://github.com/salesforce/Merlion


   
### Steps
- reference: [example](https://github.com/salesforce/Merlion/blob/main/examples/anomaly/1_AnomalyFeatures.ipynb)
- Isolation Forest: [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- VAE: https://github.com/salesforce/Merlion/blob/main/merlion/models/anomaly/vae.py

1. download market data using yfinance: download S&P 500 ('^GSPC')
1. calculate return 20 day max return (i.e. target in supervised learning problem):
   - for each date (T):
      - calculate the max price change in next 20 trading dates: price_change = (max{close price in T+1 to T+20} - {close price on T})/({close price on T})
1. use Merlion to do unsupervised anomaly detection
    1. Initializing an anomaly detection model: isolation forest, vae, ensemble
    1. Training the model
    1. Producing a series of anomaly scores with the models
    1. Visualizing the anomaly scores
1. takeaways
    - the correlation table: correlation score between the target and the anomaly score from different learning algorithms (isolation forest, vae, and ensemble of isolation forest/vae):
         - VAE shows higher correlation score compared to isolation forest and ensemble in training data while much lower correlation in testing data. 
         - This could be an indicator that VAE is overfitting the training data and shows weaker generalization capacity in testing data (unseem data in training phase).
    - Visually inspecting the target versus the anomaly scores in training and testing data
         - [plotly output](html/2_Merlion_Isoforest_VAE.html)
         - Visually, VAE seems to be doing better than isolation forest in training data but worse in testing data.


```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

from datetime import datetime, timedelta
import yfinance as yf #to download stock price data
```


```python
import matplotlib.pyplot as plt

from merlion.plot import plot_anoms
from merlion.utils import TimeSeries
```


```python
np.random.seed(5678)
```

#### download S&P 500 price data


```python
ticker = '^GSPC'
cur_data = yf.Ticker(ticker)
hist = cur_data.history(period="max")
print(ticker, hist.shape, hist.index.min())
```

    ^GSPC (19720, 7) 1927-12-30 00:00:00
    


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
    </tr>
  </tbody>
</table>
</div>



#### Merlion: Anomaly detection - unsupervised with Isolation Forest,  VAE, and ensemble with default config


```python
df.shape
```




    (5478, 9)




```python
train_ = df[['target']].iloc[:-400].copy(deep=True)
test_ = df[['target']].iloc[-400:].copy(deep=True)

train_data = TimeSeries.from_pd(train_)
test_data = TimeSeries.from_pd(test_)
```


```python
# Import models & configs
from merlion.models.anomaly.isolation_forest import IsolationForest, IsolationForestConfig
from merlion.models.anomaly.vae import VAE, VAEConfig
from merlion.models.ensemble.anomaly import DetectorEnsemble, DetectorEnsembleConfig
from merlion.post_process.threshold import AggregateAlarms

# isolation forest
iso_forest_config = IsolationForestConfig()
iso_forest_model  = IsolationForest(iso_forest_config)

# VAE
vae_config = VAEConfig()
vae_model  = VAE(vae_config)

#ensemble
en_config = DetectorEnsembleConfig(threshold=AggregateAlarms(alm_threshold=4))
en_model = DetectorEnsemble(config=en_config, models=[iso_forest_model, vae_model])
```


```python
iso_forest_train_score = iso_forest_model.train(train_data=train_data, anomaly_labels=None)
vae_train_score = vae_model.train(train_data=train_data, anomaly_labels=None)
en_train_score = en_model.train(train_data=train_data, anomaly_labels=None)
```

     |████████████████████████████████████████| 100.0% Complete, Loss 1.0673
     |████████████████████████████████████████| 100.0% Complete, Loss 1.1290
    

- Model Inference
   - model.get_anomaly_score() returns the model's raw anomaly scores,  
   - model.get_anomaly_label() returns the model's post-processed anomaly scores. The post-processing calibrates the anomaly scores to be interpretable as z-scores, and it also sparsifies them such that any nonzero values should be treated as an alert that a particular timestamp is anomalous.


```python
df_train_scores = train_.merge(iso_forest_train_score.to_pd(), left_index=True, right_index=True, how='inner')
df_train_scores = df_train_scores.merge(vae_train_score.to_pd(), left_index=True, right_index=True, how='inner')
df_train_scores = df_train_scores.merge(en_train_score.to_pd(), left_index=True, right_index=True, how='inner')
print(df_train_scores.shape, train_.shape)
df_train_scores.head(2)
```

    (5077, 4) (5078, 1)
    




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
      <th>target</th>
      <th>anom_score_x</th>
      <th>anom_score_y</th>
      <th>anom_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-02-02</th>
      <td>3.976243</td>
      <td>0.333851</td>
      <td>0.348187</td>
      <td>0.220717</td>
    </tr>
    <tr>
      <th>2000-02-03</th>
      <td>2.819712</td>
      <td>0.356733</td>
      <td>0.063608</td>
      <td>0.366427</td>
    </tr>
  </tbody>
</table>
</div>




```python
if_test_scores = iso_forest_model.get_anomaly_score(test_data)
if_test_scores_df = if_test_scores.to_pd()

if_test_labels = iso_forest_model.get_anomaly_label(test_data)
if_test_labels_df = if_test_labels.to_pd()

vae_test_scores = vae_model.get_anomaly_score(test_data)
vae_test_scores_df = vae_test_scores.to_pd()

vae_test_labels = vae_model.get_anomaly_label(test_data)
vae_test_labels_df = vae_test_labels.to_pd()

en_test_scores = en_model.get_anomaly_score(test_data)
en_test_scores_df = en_test_scores.to_pd()

en_test_labels = en_model.get_anomaly_label(test_data)
en_test_labels_df = en_test_labels.to_pd()
```


```python
df_test_scores = test_.merge(if_test_scores_df, left_index=True, right_index=True, how='inner')
df_test_scores = df_test_scores.merge(vae_test_scores_df, left_index=True, right_index=True, how='inner')
df_test_scores = df_test_scores.merge(en_test_scores_df, left_index=True, right_index=True, how='inner')

df_test_scores = df_test_scores.merge(if_test_labels_df, left_index=True, right_index=True, how='inner')
df_test_scores = df_test_scores.merge(vae_test_labels_df, left_index=True, right_index=True, how='inner')
df_test_scores = df_test_scores.merge(en_test_labels_df, left_index=True, right_index=True, how='inner')
print(test_.shape, df_test_scores.shape)
```

    (400, 1) (399, 7)
    


```python
df_train_scores.columns=['target', 'iso_forest_score', 'vae_score', 'ensemble_score']
df_test_scores.columns=['target', 'iso_forest_score', 'vae_score', 'ensemble_score', 'iso_forest_label', 'vae_label', 'ensemble_label']
```


```python
df_test_scores.head(3)
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
      <th>target</th>
      <th>iso_forest_score</th>
      <th>vae_score</th>
      <th>ensemble_score</th>
      <th>iso_forest_label</th>
      <th>vae_label</th>
      <th>ensemble_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-04-09</th>
      <td>-1.428052</td>
      <td>0.387641</td>
      <td>0.942809</td>
      <td>1.400845</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-04-13</th>
      <td>1.020781</td>
      <td>0.430124</td>
      <td>0.355629</td>
      <td>1.021738</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-04-14</th>
      <td>-1.976065</td>
      <td>0.480604</td>
      <td>1.075184</td>
      <td>1.832749</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_train_scores.corr()
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
      <th>target</th>
      <th>iso_forest_score</th>
      <th>vae_score</th>
      <th>ensemble_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>target</th>
      <td>1.000000</td>
      <td>0.661876</td>
      <td>0.828453</td>
      <td>0.710455</td>
    </tr>
    <tr>
      <th>iso_forest_score</th>
      <td>0.661876</td>
      <td>1.000000</td>
      <td>0.606901</td>
      <td>0.835899</td>
    </tr>
    <tr>
      <th>vae_score</th>
      <td>0.828453</td>
      <td>0.606901</td>
      <td>1.000000</td>
      <td>0.801360</td>
    </tr>
    <tr>
      <th>ensemble_score</th>
      <td>0.710455</td>
      <td>0.835899</td>
      <td>0.801360</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test_scores.corr()
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
      <th>target</th>
      <th>iso_forest_score</th>
      <th>vae_score</th>
      <th>ensemble_score</th>
      <th>iso_forest_label</th>
      <th>vae_label</th>
      <th>ensemble_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>target</th>
      <td>1.000000</td>
      <td>0.433776</td>
      <td>0.083479</td>
      <td>0.294011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>iso_forest_score</th>
      <td>0.433776</td>
      <td>1.000000</td>
      <td>0.027284</td>
      <td>0.688255</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>vae_score</th>
      <td>0.083479</td>
      <td>0.027284</td>
      <td>1.000000</td>
      <td>0.686652</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ensemble_score</th>
      <td>0.294011</td>
      <td>0.688255</td>
      <td>0.686652</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>iso_forest_label</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>vae_label</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ensemble_label</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test_scores['iso_forest_label'].value_counts()
```




    0.0    399
    Name: iso_forest_label, dtype: int64




```python
df_test_scores['vae_label'].value_counts()
```




    0.0    399
    Name: vae_label, dtype: int64



####  Visualizing the results

- generate graphs using plotly, display graphs inline and export graphs to a HTML file.


```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```


```python
fig_list =[]
```


```python
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])
           


fig.add_trace(go.Scatter(
                        name="target",
                        mode="lines", x=df_train_scores.index,y=df_train_scores['target']),
              secondary_y=False
             )

fig.add_trace(go.Scatter(
                        name="iso_forest_score",
                        mode="lines", x=df_train_scores.index,y=df_train_scores['iso_forest_score']),
              secondary_y=True
             )

fig.add_trace(go.Scatter(
                        name="vae_score",
                        mode="lines", x=df_train_scores.index,y=df_train_scores['vae_score']),
              secondary_y=True
             )


fig.update_layout(hovermode="x unified", 
                  title_text="Merlion Anomaly Detection with Isolation Forest and VAE - training data"
                 )



# Set y-axes titles
fig.update_yaxes(title_text="<b>target</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>anomaly score: </b> isolation forest and vae", secondary_y=True)



fig.update_xaxes(
    title_text="date", 
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()

fig_list.append(fig)
```


```python
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])



fig.add_trace(go.Scatter(
                        name="target",
                        mode="lines", x=df_test_scores.index,y=df_test_scores['target']),
              secondary_y=False
             )

fig.add_trace(go.Scatter(
                        name="iso_forest_score",
                        mode="lines", x=df_test_scores.index,y=df_test_scores['iso_forest_score']),
              secondary_y=True
             )

fig.add_trace(go.Scatter(
                        name="vae_score",
                        mode="lines", x=df_test_scores.index,y=df_test_scores['vae_score']),
              secondary_y=True
             )


fig.update_layout(hovermode="x unified", 
                  title_text="Merlion Anomaly Detection with Isolation Forest and VAE - testing data"
                 )



# Set y-axes titles
fig.update_yaxes(title_text="<b>target</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>anomaly score: </b> isolation forest and vae", secondary_y=True)



fig.update_xaxes(
    title_text="date", 
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()

fig_list.append(fig)
```


```python
fig_path = r'html/2_Merlion_Isoforest_VAE.html'
fig_list[0].write_html(fig_path)


with open(fig_path, 'a') as f:
    for fig_i in fig_list[1:]:
        f.write(fig_i.to_html(full_html=False, include_plotlyjs='cdn'))
```
