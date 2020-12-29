<h2> Quick walk through of tsfresh with financial data </h2>

`tsfresh` is a powerful feature engineering package for time series data. But there are limited examples out there on how to use it on financial time series data. 

So here we go - this is a simple and easy to follow walk through on how to use tsfresh package on stock price data. 

<h3> functions in tsfresh package</h3>

I use `roll_time_series function` to prepare data and extract_features function to generate features from `Close` price data. I also use functions under `feature_calculators` module to validate generated features. 

* `roll_time_series` prepares the raw time series data in the format that can be processed by `extract_features` function.

* `extract_features` function requires the input dataset to have at least three columns: a `‘id’` column, a `‘time’` column, and a numeric column that new features will be generated from. Note that `‘id’` column and `‘time’` column do not need to be named as `‘id’` and `‘time’`.

* `feature_calculators` module includes all the functions encapsuled in `extract_features` function.  For example, `extract_features` returns a feature with suffix `‘abs_energy’`, the values in this feature are the same as the results of using feature_calculators.abs_energy function with same input data. 

<h3> pre-process the data </h3>

Since the stock price data does not have a ‘id’ column, I create a dummy id column and set the value as a constant.



<h4> 1. load data </h4>

load csv data into a pandas dataframe.


```python
import pandas as pd
import numpy as np
import os
```


```python
df = pd.read_csv('data/^GSPC.csv')
```


```python
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
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2007-01-03</td>
      <td>1418.030029</td>
      <td>1429.420044</td>
      <td>1407.859985</td>
      <td>1416.599976</td>
      <td>1416.599976</td>
      <td>3429160000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007-01-04</td>
      <td>1416.599976</td>
      <td>1421.839966</td>
      <td>1408.430054</td>
      <td>1418.339966</td>
      <td>1418.339966</td>
      <td>3004460000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2007-01-05</td>
      <td>1418.339966</td>
      <td>1418.339966</td>
      <td>1405.750000</td>
      <td>1409.709961</td>
      <td>1409.709961</td>
      <td>2919400000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2007-01-08</td>
      <td>1409.260010</td>
      <td>1414.979980</td>
      <td>1403.969971</td>
      <td>1412.839966</td>
      <td>1412.839966</td>
      <td>2763340000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2007-01-09</td>
      <td>1412.839966</td>
      <td>1415.609985</td>
      <td>1405.420044</td>
      <td>1412.109985</td>
      <td>1412.109985</td>
      <td>3038380000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], dtype='object')



<h4> 2. pre-process data </h4>

create dummy `id` column


```python
#keep only Date and Close
df = df[['Date', 'Close']].copy(deep=True)
```


```python
#create a dummy id column
df['id']=1
```


```python
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
      <th>Date</th>
      <th>Close</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2007-01-03</td>
      <td>1416.599976</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007-01-04</td>
      <td>1418.339966</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2007-01-05</td>
      <td>1409.709961</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2007-01-08</td>
      <td>1412.839966</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2007-01-09</td>
      <td>1412.109985</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



<h4> 3. use `tsfresh` to create new features </h4>

1. prepare data using `roll_time_series`: `id` column 
1. create new features using `extract_features`


```python
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh import extract_features
from tsfresh.feature_extraction import feature_calculators
```


```python
df_rolled = roll_time_series(df, column_id="id", column_sort="Date", max_timeshift=5, min_timeshift=5, disable_progressbar=True)
```


```python
df_rolled.head()
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
      <th>Date</th>
      <th>Close</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2007-01-03</td>
      <td>1416.599976</td>
      <td>(1, 2007-01-10)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007-01-04</td>
      <td>1418.339966</td>
      <td>(1, 2007-01-10)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2007-01-05</td>
      <td>1409.709961</td>
      <td>(1, 2007-01-10)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2007-01-08</td>
      <td>1412.839966</td>
      <td>(1, 2007-01-10)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2007-01-09</td>
      <td>1412.109985</td>
      <td>(1, 2007-01-10)</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_rolled['id'].values[0]
```




    (1, '2007-01-10')




```python
df_x = df_rolled[df_rolled['id']==df_rolled['id'].values[0]]
df_x
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
      <th>Date</th>
      <th>Close</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2007-01-03</td>
      <td>1416.599976</td>
      <td>(1, 2007-01-10)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007-01-04</td>
      <td>1418.339966</td>
      <td>(1, 2007-01-10)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2007-01-05</td>
      <td>1409.709961</td>
      <td>(1, 2007-01-10)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2007-01-08</td>
      <td>1412.839966</td>
      <td>(1, 2007-01-10)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2007-01-09</td>
      <td>1412.109985</td>
      <td>(1, 2007-01-10)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2007-01-10</td>
      <td>1414.849976</td>
      <td>(1, 2007-01-10)</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_rolled[df_rolled['Date']=='2007-01-10']
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
      <th>Date</th>
      <th>Close</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>2007-01-10</td>
      <td>1414.849976</td>
      <td>(1, 2007-01-10)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2007-01-10</td>
      <td>1414.849976</td>
      <td>(1, 2007-01-11)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2007-01-10</td>
      <td>1414.849976</td>
      <td>(1, 2007-01-12)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2007-01-10</td>
      <td>1414.849976</td>
      <td>(1, 2007-01-16)</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2007-01-10</td>
      <td>1414.849976</td>
      <td>(1, 2007-01-17)</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2007-01-10</td>
      <td>1414.849976</td>
      <td>(1, 2007-01-18)</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_rolled.shape, df.shape, df_rolled.shape[0]/df.shape[0]
```




    ((19776, 3), (3301, 3), 5.990911844895487)




```python
df_rolled.shape
```




    (19776, 3)




```python
df_features = extract_features(df_rolled, column_id="id", column_sort="Date")
```

    Feature Extraction: 100%|██████████████████████████████████████████████████████████████| 20/20 [00:39<00:00,  2.00s/it]
    


```python
df_features.head()
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
      <th></th>
      <th>Close__variance_larger_than_standard_deviation</th>
      <th>Close__has_duplicate_max</th>
      <th>Close__has_duplicate_min</th>
      <th>Close__has_duplicate</th>
      <th>Close__sum_values</th>
      <th>Close__abs_energy</th>
      <th>Close__mean_abs_change</th>
      <th>Close__mean_change</th>
      <th>Close__mean_second_derivative_central</th>
      <th>Close__median</th>
      <th>...</th>
      <th>Close__fourier_entropy__bins_2</th>
      <th>Close__fourier_entropy__bins_3</th>
      <th>Close__fourier_entropy__bins_5</th>
      <th>Close__fourier_entropy__bins_10</th>
      <th>Close__fourier_entropy__bins_100</th>
      <th>Close__permutation_entropy__dimension_3__tau_1</th>
      <th>Close__permutation_entropy__dimension_4__tau_1</th>
      <th>Close__permutation_entropy__dimension_5__tau_1</th>
      <th>Close__permutation_entropy__dimension_6__tau_1</th>
      <th>Close__permutation_entropy__dimension_7__tau_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">1</th>
      <th>2007-01-10</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8484.449830</td>
      <td>1.199770e+07</td>
      <td>3.393994</td>
      <td>-0.350000</td>
      <td>0.125000</td>
      <td>1413.844971</td>
      <td>...</td>
      <td>0.562335</td>
      <td>1.039721</td>
      <td>1.039721</td>
      <td>1.039721</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>-0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2007-01-11</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8491.669800</td>
      <td>1.201821e+07</td>
      <td>4.839990</td>
      <td>1.095996</td>
      <td>2.199997</td>
      <td>1413.844971</td>
      <td>...</td>
      <td>0.562335</td>
      <td>0.562335</td>
      <td>1.039721</td>
      <td>1.039721</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>-0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2007-01-12</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8504.059814</td>
      <td>1.205351e+07</td>
      <td>4.495996</td>
      <td>4.204004</td>
      <td>0.472504</td>
      <td>1413.844971</td>
      <td>...</td>
      <td>0.562335</td>
      <td>0.562335</td>
      <td>1.039721</td>
      <td>1.039721</td>
      <td>1.039721</td>
      <td>1.039721</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>-0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2007-01-16</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8526.249877</td>
      <td>1.211656e+07</td>
      <td>4.104004</td>
      <td>3.812012</td>
      <td>0.237503</td>
      <td>1419.334961</td>
      <td>...</td>
      <td>0.562335</td>
      <td>0.562335</td>
      <td>0.562335</td>
      <td>1.039721</td>
      <td>1.386294</td>
      <td>0.562335</td>
      <td>0.636514</td>
      <td>0.693147</td>
      <td>-0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2007-01-17</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8544.029906</td>
      <td>1.216712e+07</td>
      <td>4.214014</td>
      <td>3.702002</td>
      <td>-0.502502</td>
      <td>1427.219971</td>
      <td>...</td>
      <td>0.562335</td>
      <td>1.039721</td>
      <td>1.039721</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>0.562335</td>
      <td>0.636514</td>
      <td>0.693147</td>
      <td>-0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 779 columns</p>
</div>




```python
df_features.reset_index(inplace=True)
```


```python
df_features.head()
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
      <th>level_0</th>
      <th>level_1</th>
      <th>Close__variance_larger_than_standard_deviation</th>
      <th>Close__has_duplicate_max</th>
      <th>Close__has_duplicate_min</th>
      <th>Close__has_duplicate</th>
      <th>Close__sum_values</th>
      <th>Close__abs_energy</th>
      <th>Close__mean_abs_change</th>
      <th>Close__mean_change</th>
      <th>...</th>
      <th>Close__fourier_entropy__bins_2</th>
      <th>Close__fourier_entropy__bins_3</th>
      <th>Close__fourier_entropy__bins_5</th>
      <th>Close__fourier_entropy__bins_10</th>
      <th>Close__fourier_entropy__bins_100</th>
      <th>Close__permutation_entropy__dimension_3__tau_1</th>
      <th>Close__permutation_entropy__dimension_4__tau_1</th>
      <th>Close__permutation_entropy__dimension_5__tau_1</th>
      <th>Close__permutation_entropy__dimension_6__tau_1</th>
      <th>Close__permutation_entropy__dimension_7__tau_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2007-01-10</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8484.449830</td>
      <td>1.199770e+07</td>
      <td>3.393994</td>
      <td>-0.350000</td>
      <td>...</td>
      <td>0.562335</td>
      <td>1.039721</td>
      <td>1.039721</td>
      <td>1.039721</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>-0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2007-01-11</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8491.669800</td>
      <td>1.201821e+07</td>
      <td>4.839990</td>
      <td>1.095996</td>
      <td>...</td>
      <td>0.562335</td>
      <td>0.562335</td>
      <td>1.039721</td>
      <td>1.039721</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>-0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2007-01-12</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8504.059814</td>
      <td>1.205351e+07</td>
      <td>4.495996</td>
      <td>4.204004</td>
      <td>...</td>
      <td>0.562335</td>
      <td>0.562335</td>
      <td>1.039721</td>
      <td>1.039721</td>
      <td>1.039721</td>
      <td>1.039721</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>-0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2007-01-16</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8526.249877</td>
      <td>1.211656e+07</td>
      <td>4.104004</td>
      <td>3.812012</td>
      <td>...</td>
      <td>0.562335</td>
      <td>0.562335</td>
      <td>0.562335</td>
      <td>1.039721</td>
      <td>1.386294</td>
      <td>0.562335</td>
      <td>0.636514</td>
      <td>0.693147</td>
      <td>-0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2007-01-17</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8544.029906</td>
      <td>1.216712e+07</td>
      <td>4.214014</td>
      <td>3.702002</td>
      <td>...</td>
      <td>0.562335</td>
      <td>1.039721</td>
      <td>1.039721</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>0.562335</td>
      <td>0.636514</td>
      <td>0.693147</td>
      <td>-0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 781 columns</p>
</div>




```python
df_features['level_0'].unique()
```




    array([1], dtype=int64)




```python
df_features[df_features['level_1']=='2007-01-10']
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
      <th>level_0</th>
      <th>level_1</th>
      <th>Close__variance_larger_than_standard_deviation</th>
      <th>Close__has_duplicate_max</th>
      <th>Close__has_duplicate_min</th>
      <th>Close__has_duplicate</th>
      <th>Close__sum_values</th>
      <th>Close__abs_energy</th>
      <th>Close__mean_abs_change</th>
      <th>Close__mean_change</th>
      <th>...</th>
      <th>Close__fourier_entropy__bins_2</th>
      <th>Close__fourier_entropy__bins_3</th>
      <th>Close__fourier_entropy__bins_5</th>
      <th>Close__fourier_entropy__bins_10</th>
      <th>Close__fourier_entropy__bins_100</th>
      <th>Close__permutation_entropy__dimension_3__tau_1</th>
      <th>Close__permutation_entropy__dimension_4__tau_1</th>
      <th>Close__permutation_entropy__dimension_5__tau_1</th>
      <th>Close__permutation_entropy__dimension_6__tau_1</th>
      <th>Close__permutation_entropy__dimension_7__tau_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2007-01-10</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8484.44983</td>
      <td>1.199770e+07</td>
      <td>3.393994</td>
      <td>-0.35</td>
      <td>...</td>
      <td>0.562335</td>
      <td>1.039721</td>
      <td>1.039721</td>
      <td>1.039721</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>-0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 781 columns</p>
</div>




```python
df_x.Close.var(), df_x.Close.std(), df_x.Close.sum()
```




    (9.921236996078553, 3.1497995168071498, 8484.44983)




```python
df[df['Date']<='2007-01-10']
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
      <th>Date</th>
      <th>Close</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2007-01-03</td>
      <td>1416.599976</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007-01-04</td>
      <td>1418.339966</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2007-01-05</td>
      <td>1409.709961</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2007-01-08</td>
      <td>1412.839966</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2007-01-09</td>
      <td>1412.109985</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2007-01-10</td>
      <td>1414.849976</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['Date']<='2007-01-10'].Close.var(), df[df['Date']<='2007-01-10'].Close.std(), df[df['Date']<='2007-01-10'].Close.sum()
```




    (9.921236996078553, 3.1497995168071498, 8484.44983)




```python
feature_calculators.mean_abs_change(df_x.Close), feature_calculators.mean_abs_change(df[df['Date']<='2007-01-10'].Close)
```




    (3.393994399999974, 3.393994399999974)




```python
df_x
```


```python
df_features.head()
```


```python
3301-3296
```


```python

```
