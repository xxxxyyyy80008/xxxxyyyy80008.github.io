---
title: Exploratory data analysis of AMEX Default Prediction Train Data using Dask
---

## Exploratory data analysis of AMEX Default Prediction Train Data using Dask

*Jul 9, 2022*

original file is my public notebook on Kaggle: 
- https://www.kaggle.com/code/xxxxyyyy80008/amex-default-prediction-eda-of-train-data

#### This notebook analyzes the train data in the following areas:
- general information:
    - file size, number of samples, number of features, data type of each feature.
- statistics
    - missing values: features with too many missing values should be dropped.
    - for categorical features: distribution of samples in each category.
    - numeric features: distribution of data .
      - plot out histgraph of both the original data and log tranformed data
      - plot the log transformed data will help the check if a feature will benefit from log transformation
      
      
#### key notes:
- train label file has `458,913` rows for `458,913` unique customer ID
- train data file has `5,531,451` rows for `458,913` unique customer ID
- `customer_ID` is the key to connect train label and data
- in train data:
 - there are a total of `190` columns, including key column `'customer_ID'` and date column `'S_2'`
 - a `customer_ID`  has at most 13 rows of data



```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/amex-train-20220706/train.parquet
    /kaggle/input/amex-default-prediction/sample_submission.csv
    /kaggle/input/amex-default-prediction/train_data.csv
    /kaggle/input/amex-default-prediction/test_data.csv
    /kaggle/input/amex-default-prediction/train_labels.csv
    


```python
import numpy as np
import pandas as pd
import gc
import copy
import os
import sys

from pathlib import Path
from datetime import datetime, date, time, timedelta
from dateutil import relativedelta

import pyarrow.parquet as pq
import pyarrow as pa

import dask.dataframe as dd
```


```python
import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100


import pytorch_lightning as pl
random_seed=1234
pl.seed_everything(random_seed)
```




    1234




```python
%%time
chunksize = 100
train_file = '/kaggle/input/amex-default-prediction/train_data.csv'
with pd.read_csv(train_file, chunksize=chunksize) as reader:
    for i, chunk in enumerate(reader):
        break
```

    CPU times: user 11.3 ms, sys: 7.81 ms, total: 19.1 ms
    Wall time: 18.1 ms
    


```python
chunk.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100 entries, 0 to 99
    Columns: 190 entries, customer_ID to D_145
    dtypes: float64(185), int64(1), object(4)
    memory usage: 148.6+ KB
    


```python
all_cols = chunk.columns.tolist()
#cat feats based on information from this page: https://www.kaggle.com/competitions/amex-default-prediction/data
cat_feats = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
print(len(all_cols)), print(all_cols)
print(len(cat_feats)), print(cat_feats)
```

    190
    ['customer_ID', 'S_2', 'P_2', 'D_39', 'B_1', 'B_2', 'R_1', 'S_3', 'D_41', 'B_3', 'D_42', 'D_43', 'D_44', 'B_4', 'D_45', 'B_5', 'R_2', 'D_46', 'D_47', 'D_48', 'D_49', 'B_6', 'B_7', 'B_8', 'D_50', 'D_51', 'B_9', 'R_3', 'D_52', 'P_3', 'B_10', 'D_53', 'S_5', 'B_11', 'S_6', 'D_54', 'R_4', 'S_7', 'B_12', 'S_8', 'D_55', 'D_56', 'B_13', 'R_5', 'D_58', 'S_9', 'B_14', 'D_59', 'D_60', 'D_61', 'B_15', 'S_11', 'D_62', 'D_63', 'D_64', 'D_65', 'B_16', 'B_17', 'B_18', 'B_19', 'D_66', 'B_20', 'D_68', 'S_12', 'R_6', 'S_13', 'B_21', 'D_69', 'B_22', 'D_70', 'D_71', 'D_72', 'S_15', 'B_23', 'D_73', 'P_4', 'D_74', 'D_75', 'D_76', 'B_24', 'R_7', 'D_77', 'B_25', 'B_26', 'D_78', 'D_79', 'R_8', 'R_9', 'S_16', 'D_80', 'R_10', 'R_11', 'B_27', 'D_81', 'D_82', 'S_17', 'R_12', 'B_28', 'R_13', 'D_83', 'R_14', 'R_15', 'D_84', 'R_16', 'B_29', 'B_30', 'S_18', 'D_86', 'D_87', 'R_17', 'R_18', 'D_88', 'B_31', 'S_19', 'R_19', 'B_32', 'S_20', 'R_20', 'R_21', 'B_33', 'D_89', 'R_22', 'R_23', 'D_91', 'D_92', 'D_93', 'D_94', 'R_24', 'R_25', 'D_96', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'D_102', 'D_103', 'D_104', 'D_105', 'D_106', 'D_107', 'B_36', 'B_37', 'R_26', 'R_27', 'B_38', 'D_108', 'D_109', 'D_110', 'D_111', 'B_39', 'D_112', 'B_40', 'S_27', 'D_113', 'D_114', 'D_115', 'D_116', 'D_117', 'D_118', 'D_119', 'D_120', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_126', 'D_127', 'D_128', 'D_129', 'B_41', 'B_42', 'D_130', 'D_131', 'D_132', 'D_133', 'R_28', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138', 'D_139', 'D_140', 'D_141', 'D_142', 'D_143', 'D_144', 'D_145']
    11
    ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    




    (None, None)



### load the train parquet file



```python
%%time
train_labels = pd.read_csv('/kaggle/input/amex-default-prediction/train_labels.csv')
print(train_labels.shape)
display(train_labels.head(2))
```

    (458913, 2)
    


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
      <th>customer_ID</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0000099d6bd597052cdcda90ffabf56573fe9d7c79be5f...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00000fd6641609c6ece5454664794f0340ad84dddce9a2...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    CPU times: user 638 ms, sys: 207 ms, total: 844 ms
    Wall time: 843 ms
    


```python
train_labels.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 458913 entries, 0 to 458912
    Data columns (total 2 columns):
     #   Column       Non-Null Count   Dtype 
    ---  ------       --------------   ----- 
     0   customer_ID  458913 non-null  object
     1   target       458913 non-null  int64 
    dtypes: int64(1), object(1)
    memory usage: 7.0+ MB
    


```python
train_labels['customer_ID'].nunique()
```




    458913



## calcuate the na cnt


```python
%%time
train_file = '/kaggle/input/amex-train-20220706/train.parquet'
```

    CPU times: user 3 µs, sys: 1 µs, total: 4 µs
    Wall time: 15.5 µs
    
%%time
df = pd.read_parquet(train_file, columns=['customer_ID'], engine='pyarrow')

```python
%%time
df = dd.read_parquet(train_file, columns=['customer_ID'], engine='pyarrow')
total_cnt = df.count().compute().values[0]
unique_cnt = df.compute().nunique().values[0]
print(total_cnt, unique_cnt)

del df
gc.collect()
```

    5531451 458913
    CPU times: user 4.32 s, sys: 2.24 s, total: 6.56 s
    Wall time: 6.43 s
    




    150




```python
%%time
data_types = []
for c in all_cols:
    df = dd.read_parquet(train_file, columns=[c], engine='pyarrow')
    data_types.append([c, df.dtypes.values[0], df.isna().sum().compute().values[0], df.nunique().compute().values[0]])
    
    del df
    gc.collect()
```

    CPU times: user 6min 9s, sys: 52.4 s, total: 7min 1s
    Wall time: 8min 5s
    


```python
nacnt = pd.DataFrame(data_types, columns=['feat', 'dtype', 'na_cnt', 'nunique'])
nacnt['na_pct'] = 100*nacnt['na_cnt']/total_cnt
```


```python
nacnt.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 190 entries, 108 to 95
    Data columns (total 5 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   feat     190 non-null    object 
     1   dtype    190 non-null    object 
     2   na_cnt   190 non-null    int64  
     3   nunique  190 non-null    int64  
     4   na_pct   190 non-null    float64
    dtypes: float64(1), int64(2), object(2)
    memory usage: 8.9+ KB
    


```python
%%time 
nacnt.to_csv('nacnt.csv', sep='|', index=False) #save data into a csv file for later use
```

    CPU times: user 4.03 ms, sys: 2.86 ms, total: 6.88 ms
    Wall time: 10.2 ms
    


```python
nacnt['dtype'].value_counts()
```




    float32    185
    object       4
    int32        1
    Name: dtype, dtype: int64




```python
nacnt['na_pct'].hist(bins=50)
```




    <AxesSubplot:>




    
![png](amex-default-prediction-eda-of-train-data/output_20_1.png)
    


##### base on unique number of values per feature to potential categorical feature list

- `D_87` : has only one unique value but most samples have missing values. thus this feature should not be considered as categorical features and will be dropped.
- `B_31`:  has no missing value and only two possible values. this feature will be considered as a categorical feature


```python
nacnt[nacnt['nunique']<=100]
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
      <th>feat</th>
      <th>dtype</th>
      <th>na_cnt</th>
      <th>nunique</th>
      <th>na_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53</th>
      <td>D_63</td>
      <td>object</td>
      <td>0</td>
      <td>6</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>54</th>
      <td>D_64</td>
      <td>object</td>
      <td>217442</td>
      <td>4</td>
      <td>3.931012</td>
    </tr>
    <tr>
      <th>60</th>
      <td>D_66</td>
      <td>float32</td>
      <td>4908097</td>
      <td>2</td>
      <td>88.730733</td>
    </tr>
    <tr>
      <th>62</th>
      <td>D_68</td>
      <td>float32</td>
      <td>216503</td>
      <td>7</td>
      <td>3.914036</td>
    </tr>
    <tr>
      <th>105</th>
      <td>B_30</td>
      <td>float32</td>
      <td>2016</td>
      <td>3</td>
      <td>0.036446</td>
    </tr>
    <tr>
      <th>108</th>
      <td>D_87</td>
      <td>float32</td>
      <td>5527586</td>
      <td>1</td>
      <td>99.930127</td>
    </tr>
    <tr>
      <th>112</th>
      <td>B_31</td>
      <td>int32</td>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>145</th>
      <td>B_38</td>
      <td>float32</td>
      <td>2016</td>
      <td>7</td>
      <td>0.036446</td>
    </tr>
    <tr>
      <th>155</th>
      <td>D_114</td>
      <td>float32</td>
      <td>176716</td>
      <td>2</td>
      <td>3.194749</td>
    </tr>
    <tr>
      <th>157</th>
      <td>D_116</td>
      <td>float32</td>
      <td>176716</td>
      <td>2</td>
      <td>3.194749</td>
    </tr>
    <tr>
      <th>158</th>
      <td>D_117</td>
      <td>float32</td>
      <td>176716</td>
      <td>7</td>
      <td>3.194749</td>
    </tr>
    <tr>
      <th>161</th>
      <td>D_120</td>
      <td>float32</td>
      <td>176716</td>
      <td>2</td>
      <td>3.194749</td>
    </tr>
    <tr>
      <th>167</th>
      <td>D_126</td>
      <td>float32</td>
      <td>116816</td>
      <td>3</td>
      <td>2.111851</td>
    </tr>
  </tbody>
</table>
</div>




```python
potential_cat_feats = nacnt[nacnt['nunique']<=100]['feat'].values.tolist()
print(potential_cat_feats)
print(set(potential_cat_feats)-set(cat_feats))
```

    ['D_63', 'D_64', 'D_66', 'D_68', 'B_30', 'D_87', 'B_31', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']
    {'B_31', 'D_87'}
    

#### missing values

- features with too much missing values will be dropped
- `>=80%` samples missing value: 23 features
 - `missing80 = ['D_87', 'D_88', 'D_108', 'D_111', 'D_110', 'B_39', 'D_73', 'B_42', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138', 'R_9', 'B_29', 'D_106', 'D_132', 'D_49', 'R_26', 'D_76', 'D_66', 'D_42', 'D_142']`
- `>=50%` samples missing value: 30 features
 - `missing50 = ['D_87', 'D_88', 'D_108', 'D_111', 'D_110', 'B_39', 'D_73', 'B_42', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138', 'R_9', 'B_29', 'D_106', 'D_132', 'D_49', 'R_26', 'D_76', 'D_66', 'D_42', 'D_142', 'D_53', 'D_82', 'D_50', 'B_17', 'D_105', 'D_56', 'S_9']`
- `>=20%` samples missing value: 34 features
 - `missing20 = ['D_87', 'D_88', 'D_108', 'D_111', 'D_110', 'B_39', 'D_73', 'B_42', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138', 'R_9', 'B_29', 'D_106', 'D_132', 'D_49', 'R_26', 'D_76', 'D_66', 'D_42', 'D_142', 'D_53', 'D_82', 'D_50', 'B_17', 'D_105', 'D_56', 'S_9', 'D_77', 'D_43', 'S_27', 'D_46']`
- `>=10%` samples missing value: 39 fatures
 - `missing10 = ['D_87', 'D_88', 'D_108', 'D_111', 'D_110', 'B_39', 'D_73', 'B_42', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138', 'R_9', 'B_29', 'D_106', 'D_132', 'D_49', 'R_26', 'D_76', 'D_66', 'D_42', 'D_142', 'D_53', 'D_82', 'D_50', 'B_17', 'D_105', 'D_56', 'S_9', 'D_77', 'D_43', 'S_27', 'D_46', 'S_7', 'S_3', 'D_62', 'D_48', 'D_61']`


```python
nacnt.sort_values(by='na_pct', ascending=False, inplace=True)
```


```python
nacnt.head(20)
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
      <th>feat</th>
      <th>dtype</th>
      <th>na_cnt</th>
      <th>nunique</th>
      <th>na_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>108</th>
      <td>D_87</td>
      <td>float32</td>
      <td>5527586</td>
      <td>1</td>
      <td>99.930127</td>
    </tr>
    <tr>
      <th>111</th>
      <td>D_88</td>
      <td>float32</td>
      <td>5525447</td>
      <td>6004</td>
      <td>99.891457</td>
    </tr>
    <tr>
      <th>146</th>
      <td>D_108</td>
      <td>float32</td>
      <td>5502513</td>
      <td>28902</td>
      <td>99.476846</td>
    </tr>
    <tr>
      <th>149</th>
      <td>D_111</td>
      <td>float32</td>
      <td>5500117</td>
      <td>27968</td>
      <td>99.433530</td>
    </tr>
    <tr>
      <th>148</th>
      <td>D_110</td>
      <td>float32</td>
      <td>5500117</td>
      <td>30406</td>
      <td>99.433530</td>
    </tr>
    <tr>
      <th>150</th>
      <td>B_39</td>
      <td>float32</td>
      <td>5497819</td>
      <td>33557</td>
      <td>99.391986</td>
    </tr>
    <tr>
      <th>74</th>
      <td>D_73</td>
      <td>float32</td>
      <td>5475595</td>
      <td>55802</td>
      <td>98.990211</td>
    </tr>
    <tr>
      <th>172</th>
      <td>B_42</td>
      <td>float32</td>
      <td>5459973</td>
      <td>71434</td>
      <td>98.707789</td>
    </tr>
    <tr>
      <th>178</th>
      <td>D_134</td>
      <td>float32</td>
      <td>5336752</td>
      <td>192523</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>179</th>
      <td>D_135</td>
      <td>float32</td>
      <td>5336752</td>
      <td>193754</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>180</th>
      <td>D_136</td>
      <td>float32</td>
      <td>5336752</td>
      <td>175899</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>181</th>
      <td>D_137</td>
      <td>float32</td>
      <td>5336752</td>
      <td>193810</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>182</th>
      <td>D_138</td>
      <td>float32</td>
      <td>5336752</td>
      <td>186987</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>87</th>
      <td>R_9</td>
      <td>float32</td>
      <td>5218918</td>
      <td>278226</td>
      <td>94.349891</td>
    </tr>
    <tr>
      <th>104</th>
      <td>B_29</td>
      <td>float32</td>
      <td>5150035</td>
      <td>378319</td>
      <td>93.104594</td>
    </tr>
    <tr>
      <th>139</th>
      <td>D_106</td>
      <td>float32</td>
      <td>4990102</td>
      <td>526748</td>
      <td>90.213255</td>
    </tr>
    <tr>
      <th>175</th>
      <td>D_132</td>
      <td>float32</td>
      <td>4988874</td>
      <td>538748</td>
      <td>90.191055</td>
    </tr>
    <tr>
      <th>20</th>
      <td>D_49</td>
      <td>float32</td>
      <td>4985917</td>
      <td>540617</td>
      <td>90.137597</td>
    </tr>
    <tr>
      <th>143</th>
      <td>R_26</td>
      <td>float32</td>
      <td>4922146</td>
      <td>601763</td>
      <td>88.984717</td>
    </tr>
    <tr>
      <th>78</th>
      <td>D_76</td>
      <td>float32</td>
      <td>4908954</td>
      <td>619107</td>
      <td>88.746226</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(nacnt[nacnt['na_pct']>=80].shape)
missing80 = nacnt[nacnt['na_pct']>=80]['feat'].values.tolist()
print('number of features with >=80% samples missing values: ', len(missing80))
print(missing80)
nacnt[nacnt['na_pct']>=80]
```

    (23, 5)
    number of features with >=80% samples missing values:  23
    ['D_87', 'D_88', 'D_108', 'D_111', 'D_110', 'B_39', 'D_73', 'B_42', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138', 'R_9', 'B_29', 'D_106', 'D_132', 'D_49', 'R_26', 'D_76', 'D_66', 'D_42', 'D_142']
    




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
      <th>feat</th>
      <th>dtype</th>
      <th>na_cnt</th>
      <th>nunique</th>
      <th>na_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>108</th>
      <td>D_87</td>
      <td>float32</td>
      <td>5527586</td>
      <td>1</td>
      <td>99.930127</td>
    </tr>
    <tr>
      <th>111</th>
      <td>D_88</td>
      <td>float32</td>
      <td>5525447</td>
      <td>6004</td>
      <td>99.891457</td>
    </tr>
    <tr>
      <th>146</th>
      <td>D_108</td>
      <td>float32</td>
      <td>5502513</td>
      <td>28902</td>
      <td>99.476846</td>
    </tr>
    <tr>
      <th>149</th>
      <td>D_111</td>
      <td>float32</td>
      <td>5500117</td>
      <td>27968</td>
      <td>99.433530</td>
    </tr>
    <tr>
      <th>148</th>
      <td>D_110</td>
      <td>float32</td>
      <td>5500117</td>
      <td>30406</td>
      <td>99.433530</td>
    </tr>
    <tr>
      <th>150</th>
      <td>B_39</td>
      <td>float32</td>
      <td>5497819</td>
      <td>33557</td>
      <td>99.391986</td>
    </tr>
    <tr>
      <th>74</th>
      <td>D_73</td>
      <td>float32</td>
      <td>5475595</td>
      <td>55802</td>
      <td>98.990211</td>
    </tr>
    <tr>
      <th>172</th>
      <td>B_42</td>
      <td>float32</td>
      <td>5459973</td>
      <td>71434</td>
      <td>98.707789</td>
    </tr>
    <tr>
      <th>178</th>
      <td>D_134</td>
      <td>float32</td>
      <td>5336752</td>
      <td>192523</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>179</th>
      <td>D_135</td>
      <td>float32</td>
      <td>5336752</td>
      <td>193754</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>180</th>
      <td>D_136</td>
      <td>float32</td>
      <td>5336752</td>
      <td>175899</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>181</th>
      <td>D_137</td>
      <td>float32</td>
      <td>5336752</td>
      <td>193810</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>182</th>
      <td>D_138</td>
      <td>float32</td>
      <td>5336752</td>
      <td>186987</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>87</th>
      <td>R_9</td>
      <td>float32</td>
      <td>5218918</td>
      <td>278226</td>
      <td>94.349891</td>
    </tr>
    <tr>
      <th>104</th>
      <td>B_29</td>
      <td>float32</td>
      <td>5150035</td>
      <td>378319</td>
      <td>93.104594</td>
    </tr>
    <tr>
      <th>139</th>
      <td>D_106</td>
      <td>float32</td>
      <td>4990102</td>
      <td>526748</td>
      <td>90.213255</td>
    </tr>
    <tr>
      <th>175</th>
      <td>D_132</td>
      <td>float32</td>
      <td>4988874</td>
      <td>538748</td>
      <td>90.191055</td>
    </tr>
    <tr>
      <th>20</th>
      <td>D_49</td>
      <td>float32</td>
      <td>4985917</td>
      <td>540617</td>
      <td>90.137597</td>
    </tr>
    <tr>
      <th>143</th>
      <td>R_26</td>
      <td>float32</td>
      <td>4922146</td>
      <td>601763</td>
      <td>88.984717</td>
    </tr>
    <tr>
      <th>78</th>
      <td>D_76</td>
      <td>float32</td>
      <td>4908954</td>
      <td>619107</td>
      <td>88.746226</td>
    </tr>
    <tr>
      <th>60</th>
      <td>D_66</td>
      <td>float32</td>
      <td>4908097</td>
      <td>2</td>
      <td>88.730733</td>
    </tr>
    <tr>
      <th>10</th>
      <td>D_42</td>
      <td>float32</td>
      <td>4740137</td>
      <td>785339</td>
      <td>85.694278</td>
    </tr>
    <tr>
      <th>186</th>
      <td>D_142</td>
      <td>float32</td>
      <td>4587043</td>
      <td>926344</td>
      <td>82.926577</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(nacnt[nacnt['na_pct']>=50].shape)
missing50 = nacnt[nacnt['na_pct']>=50]['feat'].values.tolist()
print('number of features with >=50% samples missing values: ', len(missing50))
print(missing50)
nacnt[nacnt['na_pct']>=50]
```

    (30, 5)
    number of features with >=50% samples missing values:  30
    ['D_87', 'D_88', 'D_108', 'D_111', 'D_110', 'B_39', 'D_73', 'B_42', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138', 'R_9', 'B_29', 'D_106', 'D_132', 'D_49', 'R_26', 'D_76', 'D_66', 'D_42', 'D_142', 'D_53', 'D_82', 'D_50', 'B_17', 'D_105', 'D_56', 'S_9']
    




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
      <th>feat</th>
      <th>dtype</th>
      <th>na_cnt</th>
      <th>nunique</th>
      <th>na_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>108</th>
      <td>D_87</td>
      <td>float32</td>
      <td>5527586</td>
      <td>1</td>
      <td>99.930127</td>
    </tr>
    <tr>
      <th>111</th>
      <td>D_88</td>
      <td>float32</td>
      <td>5525447</td>
      <td>6004</td>
      <td>99.891457</td>
    </tr>
    <tr>
      <th>146</th>
      <td>D_108</td>
      <td>float32</td>
      <td>5502513</td>
      <td>28902</td>
      <td>99.476846</td>
    </tr>
    <tr>
      <th>149</th>
      <td>D_111</td>
      <td>float32</td>
      <td>5500117</td>
      <td>27968</td>
      <td>99.433530</td>
    </tr>
    <tr>
      <th>148</th>
      <td>D_110</td>
      <td>float32</td>
      <td>5500117</td>
      <td>30406</td>
      <td>99.433530</td>
    </tr>
    <tr>
      <th>150</th>
      <td>B_39</td>
      <td>float32</td>
      <td>5497819</td>
      <td>33557</td>
      <td>99.391986</td>
    </tr>
    <tr>
      <th>74</th>
      <td>D_73</td>
      <td>float32</td>
      <td>5475595</td>
      <td>55802</td>
      <td>98.990211</td>
    </tr>
    <tr>
      <th>172</th>
      <td>B_42</td>
      <td>float32</td>
      <td>5459973</td>
      <td>71434</td>
      <td>98.707789</td>
    </tr>
    <tr>
      <th>178</th>
      <td>D_134</td>
      <td>float32</td>
      <td>5336752</td>
      <td>192523</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>179</th>
      <td>D_135</td>
      <td>float32</td>
      <td>5336752</td>
      <td>193754</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>180</th>
      <td>D_136</td>
      <td>float32</td>
      <td>5336752</td>
      <td>175899</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>181</th>
      <td>D_137</td>
      <td>float32</td>
      <td>5336752</td>
      <td>193810</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>182</th>
      <td>D_138</td>
      <td>float32</td>
      <td>5336752</td>
      <td>186987</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>87</th>
      <td>R_9</td>
      <td>float32</td>
      <td>5218918</td>
      <td>278226</td>
      <td>94.349891</td>
    </tr>
    <tr>
      <th>104</th>
      <td>B_29</td>
      <td>float32</td>
      <td>5150035</td>
      <td>378319</td>
      <td>93.104594</td>
    </tr>
    <tr>
      <th>139</th>
      <td>D_106</td>
      <td>float32</td>
      <td>4990102</td>
      <td>526748</td>
      <td>90.213255</td>
    </tr>
    <tr>
      <th>175</th>
      <td>D_132</td>
      <td>float32</td>
      <td>4988874</td>
      <td>538748</td>
      <td>90.191055</td>
    </tr>
    <tr>
      <th>20</th>
      <td>D_49</td>
      <td>float32</td>
      <td>4985917</td>
      <td>540617</td>
      <td>90.137597</td>
    </tr>
    <tr>
      <th>143</th>
      <td>R_26</td>
      <td>float32</td>
      <td>4922146</td>
      <td>601763</td>
      <td>88.984717</td>
    </tr>
    <tr>
      <th>78</th>
      <td>D_76</td>
      <td>float32</td>
      <td>4908954</td>
      <td>619107</td>
      <td>88.746226</td>
    </tr>
    <tr>
      <th>60</th>
      <td>D_66</td>
      <td>float32</td>
      <td>4908097</td>
      <td>2</td>
      <td>88.730733</td>
    </tr>
    <tr>
      <th>10</th>
      <td>D_42</td>
      <td>float32</td>
      <td>4740137</td>
      <td>785339</td>
      <td>85.694278</td>
    </tr>
    <tr>
      <th>186</th>
      <td>D_142</td>
      <td>float32</td>
      <td>4587043</td>
      <td>926344</td>
      <td>82.926577</td>
    </tr>
    <tr>
      <th>31</th>
      <td>D_53</td>
      <td>float32</td>
      <td>4084585</td>
      <td>1428989</td>
      <td>73.842921</td>
    </tr>
    <tr>
      <th>94</th>
      <td>D_82</td>
      <td>float32</td>
      <td>4058614</td>
      <td>377661</td>
      <td>73.373406</td>
    </tr>
    <tr>
      <th>24</th>
      <td>D_50</td>
      <td>float32</td>
      <td>3142402</td>
      <td>2306378</td>
      <td>56.809723</td>
    </tr>
    <tr>
      <th>57</th>
      <td>B_17</td>
      <td>float32</td>
      <td>3137598</td>
      <td>1592707</td>
      <td>56.722874</td>
    </tr>
    <tr>
      <th>138</th>
      <td>D_105</td>
      <td>float32</td>
      <td>3021431</td>
      <td>2414778</td>
      <td>54.622756</td>
    </tr>
    <tr>
      <th>41</th>
      <td>D_56</td>
      <td>float32</td>
      <td>2990943</td>
      <td>2446769</td>
      <td>54.071581</td>
    </tr>
    <tr>
      <th>45</th>
      <td>S_9</td>
      <td>float32</td>
      <td>2933643</td>
      <td>2531874</td>
      <td>53.035686</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(nacnt[nacnt['na_pct']>=20].shape)
missing20 = nacnt[nacnt['na_pct']>=20]['feat'].values.tolist()
print('number of features with >=20% samples missing values: ', len(missing20))
print(missing20)
nacnt[nacnt['na_pct']>=20]
```

    (34, 5)
    number of features with >=20% samples missing values:  34
    ['D_87', 'D_88', 'D_108', 'D_111', 'D_110', 'B_39', 'D_73', 'B_42', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138', 'R_9', 'B_29', 'D_106', 'D_132', 'D_49', 'R_26', 'D_76', 'D_66', 'D_42', 'D_142', 'D_53', 'D_82', 'D_50', 'B_17', 'D_105', 'D_56', 'S_9', 'D_77', 'D_43', 'S_27', 'D_46']
    




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
      <th>feat</th>
      <th>dtype</th>
      <th>na_cnt</th>
      <th>nunique</th>
      <th>na_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>108</th>
      <td>D_87</td>
      <td>float32</td>
      <td>5527586</td>
      <td>1</td>
      <td>99.930127</td>
    </tr>
    <tr>
      <th>111</th>
      <td>D_88</td>
      <td>float32</td>
      <td>5525447</td>
      <td>6004</td>
      <td>99.891457</td>
    </tr>
    <tr>
      <th>146</th>
      <td>D_108</td>
      <td>float32</td>
      <td>5502513</td>
      <td>28902</td>
      <td>99.476846</td>
    </tr>
    <tr>
      <th>149</th>
      <td>D_111</td>
      <td>float32</td>
      <td>5500117</td>
      <td>27968</td>
      <td>99.433530</td>
    </tr>
    <tr>
      <th>148</th>
      <td>D_110</td>
      <td>float32</td>
      <td>5500117</td>
      <td>30406</td>
      <td>99.433530</td>
    </tr>
    <tr>
      <th>150</th>
      <td>B_39</td>
      <td>float32</td>
      <td>5497819</td>
      <td>33557</td>
      <td>99.391986</td>
    </tr>
    <tr>
      <th>74</th>
      <td>D_73</td>
      <td>float32</td>
      <td>5475595</td>
      <td>55802</td>
      <td>98.990211</td>
    </tr>
    <tr>
      <th>172</th>
      <td>B_42</td>
      <td>float32</td>
      <td>5459973</td>
      <td>71434</td>
      <td>98.707789</td>
    </tr>
    <tr>
      <th>178</th>
      <td>D_134</td>
      <td>float32</td>
      <td>5336752</td>
      <td>192523</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>179</th>
      <td>D_135</td>
      <td>float32</td>
      <td>5336752</td>
      <td>193754</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>180</th>
      <td>D_136</td>
      <td>float32</td>
      <td>5336752</td>
      <td>175899</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>181</th>
      <td>D_137</td>
      <td>float32</td>
      <td>5336752</td>
      <td>193810</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>182</th>
      <td>D_138</td>
      <td>float32</td>
      <td>5336752</td>
      <td>186987</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>87</th>
      <td>R_9</td>
      <td>float32</td>
      <td>5218918</td>
      <td>278226</td>
      <td>94.349891</td>
    </tr>
    <tr>
      <th>104</th>
      <td>B_29</td>
      <td>float32</td>
      <td>5150035</td>
      <td>378319</td>
      <td>93.104594</td>
    </tr>
    <tr>
      <th>139</th>
      <td>D_106</td>
      <td>float32</td>
      <td>4990102</td>
      <td>526748</td>
      <td>90.213255</td>
    </tr>
    <tr>
      <th>175</th>
      <td>D_132</td>
      <td>float32</td>
      <td>4988874</td>
      <td>538748</td>
      <td>90.191055</td>
    </tr>
    <tr>
      <th>20</th>
      <td>D_49</td>
      <td>float32</td>
      <td>4985917</td>
      <td>540617</td>
      <td>90.137597</td>
    </tr>
    <tr>
      <th>143</th>
      <td>R_26</td>
      <td>float32</td>
      <td>4922146</td>
      <td>601763</td>
      <td>88.984717</td>
    </tr>
    <tr>
      <th>78</th>
      <td>D_76</td>
      <td>float32</td>
      <td>4908954</td>
      <td>619107</td>
      <td>88.746226</td>
    </tr>
    <tr>
      <th>60</th>
      <td>D_66</td>
      <td>float32</td>
      <td>4908097</td>
      <td>2</td>
      <td>88.730733</td>
    </tr>
    <tr>
      <th>10</th>
      <td>D_42</td>
      <td>float32</td>
      <td>4740137</td>
      <td>785339</td>
      <td>85.694278</td>
    </tr>
    <tr>
      <th>186</th>
      <td>D_142</td>
      <td>float32</td>
      <td>4587043</td>
      <td>926344</td>
      <td>82.926577</td>
    </tr>
    <tr>
      <th>31</th>
      <td>D_53</td>
      <td>float32</td>
      <td>4084585</td>
      <td>1428989</td>
      <td>73.842921</td>
    </tr>
    <tr>
      <th>94</th>
      <td>D_82</td>
      <td>float32</td>
      <td>4058614</td>
      <td>377661</td>
      <td>73.373406</td>
    </tr>
    <tr>
      <th>24</th>
      <td>D_50</td>
      <td>float32</td>
      <td>3142402</td>
      <td>2306378</td>
      <td>56.809723</td>
    </tr>
    <tr>
      <th>57</th>
      <td>B_17</td>
      <td>float32</td>
      <td>3137598</td>
      <td>1592707</td>
      <td>56.722874</td>
    </tr>
    <tr>
      <th>138</th>
      <td>D_105</td>
      <td>float32</td>
      <td>3021431</td>
      <td>2414778</td>
      <td>54.622756</td>
    </tr>
    <tr>
      <th>41</th>
      <td>D_56</td>
      <td>float32</td>
      <td>2990943</td>
      <td>2446769</td>
      <td>54.071581</td>
    </tr>
    <tr>
      <th>45</th>
      <td>S_9</td>
      <td>float32</td>
      <td>2933643</td>
      <td>2531874</td>
      <td>53.035686</td>
    </tr>
    <tr>
      <th>81</th>
      <td>D_77</td>
      <td>float32</td>
      <td>2513912</td>
      <td>2859646</td>
      <td>45.447605</td>
    </tr>
    <tr>
      <th>11</th>
      <td>D_43</td>
      <td>float32</td>
      <td>1658396</td>
      <td>3708893</td>
      <td>29.981211</td>
    </tr>
    <tr>
      <th>153</th>
      <td>S_27</td>
      <td>float32</td>
      <td>1400935</td>
      <td>3885609</td>
      <td>25.326718</td>
    </tr>
    <tr>
      <th>17</th>
      <td>D_46</td>
      <td>float32</td>
      <td>1211699</td>
      <td>3388402</td>
      <td>21.905627</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(nacnt[nacnt['na_pct']>=10].shape)
missing10 = nacnt[nacnt['na_pct']>=10]['feat'].values.tolist()
print('number of features with >=10% samples missing values: ', len(missing10))
print(missing10)
nacnt[nacnt['na_pct']>=10]
```

    (39, 5)
    number of features with >=10% samples missing values:  39
    ['D_87', 'D_88', 'D_108', 'D_111', 'D_110', 'B_39', 'D_73', 'B_42', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138', 'R_9', 'B_29', 'D_106', 'D_132', 'D_49', 'R_26', 'D_76', 'D_66', 'D_42', 'D_142', 'D_53', 'D_82', 'D_50', 'B_17', 'D_105', 'D_56', 'S_9', 'D_77', 'D_43', 'S_27', 'D_46', 'S_7', 'S_3', 'D_62', 'D_48', 'D_61']
    




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
      <th>feat</th>
      <th>dtype</th>
      <th>na_cnt</th>
      <th>nunique</th>
      <th>na_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>108</th>
      <td>D_87</td>
      <td>float32</td>
      <td>5527586</td>
      <td>1</td>
      <td>99.930127</td>
    </tr>
    <tr>
      <th>111</th>
      <td>D_88</td>
      <td>float32</td>
      <td>5525447</td>
      <td>6004</td>
      <td>99.891457</td>
    </tr>
    <tr>
      <th>146</th>
      <td>D_108</td>
      <td>float32</td>
      <td>5502513</td>
      <td>28902</td>
      <td>99.476846</td>
    </tr>
    <tr>
      <th>149</th>
      <td>D_111</td>
      <td>float32</td>
      <td>5500117</td>
      <td>27968</td>
      <td>99.433530</td>
    </tr>
    <tr>
      <th>148</th>
      <td>D_110</td>
      <td>float32</td>
      <td>5500117</td>
      <td>30406</td>
      <td>99.433530</td>
    </tr>
    <tr>
      <th>150</th>
      <td>B_39</td>
      <td>float32</td>
      <td>5497819</td>
      <td>33557</td>
      <td>99.391986</td>
    </tr>
    <tr>
      <th>74</th>
      <td>D_73</td>
      <td>float32</td>
      <td>5475595</td>
      <td>55802</td>
      <td>98.990211</td>
    </tr>
    <tr>
      <th>172</th>
      <td>B_42</td>
      <td>float32</td>
      <td>5459973</td>
      <td>71434</td>
      <td>98.707789</td>
    </tr>
    <tr>
      <th>178</th>
      <td>D_134</td>
      <td>float32</td>
      <td>5336752</td>
      <td>192523</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>179</th>
      <td>D_135</td>
      <td>float32</td>
      <td>5336752</td>
      <td>193754</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>180</th>
      <td>D_136</td>
      <td>float32</td>
      <td>5336752</td>
      <td>175899</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>181</th>
      <td>D_137</td>
      <td>float32</td>
      <td>5336752</td>
      <td>193810</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>182</th>
      <td>D_138</td>
      <td>float32</td>
      <td>5336752</td>
      <td>186987</td>
      <td>96.480146</td>
    </tr>
    <tr>
      <th>87</th>
      <td>R_9</td>
      <td>float32</td>
      <td>5218918</td>
      <td>278226</td>
      <td>94.349891</td>
    </tr>
    <tr>
      <th>104</th>
      <td>B_29</td>
      <td>float32</td>
      <td>5150035</td>
      <td>378319</td>
      <td>93.104594</td>
    </tr>
    <tr>
      <th>139</th>
      <td>D_106</td>
      <td>float32</td>
      <td>4990102</td>
      <td>526748</td>
      <td>90.213255</td>
    </tr>
    <tr>
      <th>175</th>
      <td>D_132</td>
      <td>float32</td>
      <td>4988874</td>
      <td>538748</td>
      <td>90.191055</td>
    </tr>
    <tr>
      <th>20</th>
      <td>D_49</td>
      <td>float32</td>
      <td>4985917</td>
      <td>540617</td>
      <td>90.137597</td>
    </tr>
    <tr>
      <th>143</th>
      <td>R_26</td>
      <td>float32</td>
      <td>4922146</td>
      <td>601763</td>
      <td>88.984717</td>
    </tr>
    <tr>
      <th>78</th>
      <td>D_76</td>
      <td>float32</td>
      <td>4908954</td>
      <td>619107</td>
      <td>88.746226</td>
    </tr>
    <tr>
      <th>60</th>
      <td>D_66</td>
      <td>float32</td>
      <td>4908097</td>
      <td>2</td>
      <td>88.730733</td>
    </tr>
    <tr>
      <th>10</th>
      <td>D_42</td>
      <td>float32</td>
      <td>4740137</td>
      <td>785339</td>
      <td>85.694278</td>
    </tr>
    <tr>
      <th>186</th>
      <td>D_142</td>
      <td>float32</td>
      <td>4587043</td>
      <td>926344</td>
      <td>82.926577</td>
    </tr>
    <tr>
      <th>31</th>
      <td>D_53</td>
      <td>float32</td>
      <td>4084585</td>
      <td>1428989</td>
      <td>73.842921</td>
    </tr>
    <tr>
      <th>94</th>
      <td>D_82</td>
      <td>float32</td>
      <td>4058614</td>
      <td>377661</td>
      <td>73.373406</td>
    </tr>
    <tr>
      <th>24</th>
      <td>D_50</td>
      <td>float32</td>
      <td>3142402</td>
      <td>2306378</td>
      <td>56.809723</td>
    </tr>
    <tr>
      <th>57</th>
      <td>B_17</td>
      <td>float32</td>
      <td>3137598</td>
      <td>1592707</td>
      <td>56.722874</td>
    </tr>
    <tr>
      <th>138</th>
      <td>D_105</td>
      <td>float32</td>
      <td>3021431</td>
      <td>2414778</td>
      <td>54.622756</td>
    </tr>
    <tr>
      <th>41</th>
      <td>D_56</td>
      <td>float32</td>
      <td>2990943</td>
      <td>2446769</td>
      <td>54.071581</td>
    </tr>
    <tr>
      <th>45</th>
      <td>S_9</td>
      <td>float32</td>
      <td>2933643</td>
      <td>2531874</td>
      <td>53.035686</td>
    </tr>
    <tr>
      <th>81</th>
      <td>D_77</td>
      <td>float32</td>
      <td>2513912</td>
      <td>2859646</td>
      <td>45.447605</td>
    </tr>
    <tr>
      <th>11</th>
      <td>D_43</td>
      <td>float32</td>
      <td>1658396</td>
      <td>3708893</td>
      <td>29.981211</td>
    </tr>
    <tr>
      <th>153</th>
      <td>S_27</td>
      <td>float32</td>
      <td>1400935</td>
      <td>3885609</td>
      <td>25.326718</td>
    </tr>
    <tr>
      <th>17</th>
      <td>D_46</td>
      <td>float32</td>
      <td>1211699</td>
      <td>3388402</td>
      <td>21.905627</td>
    </tr>
    <tr>
      <th>37</th>
      <td>S_7</td>
      <td>float32</td>
      <td>1020544</td>
      <td>4185390</td>
      <td>18.449843</td>
    </tr>
    <tr>
      <th>7</th>
      <td>S_3</td>
      <td>float32</td>
      <td>1020544</td>
      <td>4004312</td>
      <td>18.449843</td>
    </tr>
    <tr>
      <th>52</th>
      <td>D_62</td>
      <td>float32</td>
      <td>758161</td>
      <td>4519358</td>
      <td>13.706367</td>
    </tr>
    <tr>
      <th>19</th>
      <td>D_48</td>
      <td>float32</td>
      <td>718725</td>
      <td>4534865</td>
      <td>12.993426</td>
    </tr>
    <tr>
      <th>49</th>
      <td>D_61</td>
      <td>float32</td>
      <td>598052</td>
      <td>4529133</td>
      <td>10.811847</td>
    </tr>
  </tbody>
</table>
</div>



## categorical features


```python
cat_feats = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
```


```python
%%time
for c in cat_feats:
    df = dd.read_parquet(train_file, columns=[c], engine='pyarrow')
    print(c, '-'*100)
    print(df[c].nunique().compute())
    display(df[c].value_counts().compute().sort_index())
    del df
    gc.collect()
```

    B_30 ----------------------------------------------------------------------------------------------------
    3
    


    0.0    4710663
    1.0     763955
    2.0      54817
    Name: B_30, dtype: int64


    B_38 ----------------------------------------------------------------------------------------------------
    7
    


    1.0    1160047
    2.0    1953232
    3.0    1255315
    4.0     294917
    5.0     444856
    6.0     162040
    7.0     259028
    Name: B_38, dtype: int64


    D_114 ----------------------------------------------------------------------------------------------------
    2
    


    0.0    2038257
    1.0    3316478
    Name: D_114, dtype: int64


    D_116 ----------------------------------------------------------------------------------------------------
    2
    


    0.0    5348109
    1.0       6626
    Name: D_116, dtype: int64


    D_117 ----------------------------------------------------------------------------------------------------
    7
    


    -1.0    1456084
     1.0     122967
     2.0     666808
     3.0    1166400
     4.0    1138666
     5.0     459290
     6.0     344520
    Name: D_117, dtype: int64


    D_120 ----------------------------------------------------------------------------------------------------
    2
    


    0.0    4729723
    1.0     625012
    Name: D_120, dtype: int64


    D_126 ----------------------------------------------------------------------------------------------------
    3
    


    -1.0     260898
     0.0     891323
     1.0    4262414
    Name: D_126, dtype: int64


    D_63 ----------------------------------------------------------------------------------------------------
    6
    


    CL     438390
    CO    4119621
    CR     930133
    XL       6965
    XM      10556
    XZ      25786
    Name: D_63, dtype: int64


    D_64 ----------------------------------------------------------------------------------------------------
    4
    


    -1      37205
    O     2913244
    R      840112
    U     1523448
    Name: D_64, dtype: int64


    D_66 ----------------------------------------------------------------------------------------------------
    2
    


    0.0      6288
    1.0    617066
    Name: D_66, dtype: int64


    D_68 ----------------------------------------------------------------------------------------------------
    7
    


    0.0      15925
    1.0     133122
    2.0     220111
    3.0     484442
    4.0     477187
    5.0    1201706
    6.0    2782455
    Name: D_68, dtype: int64


    CPU times: user 8.76 s, sys: 1.13 s, total: 9.89 s
    Wall time: 9.76 s
    

### object features

- `customer_ID`: customer id. 
- `S_2`: date
- `D_63, D_64`: categorical features



```python
print(nacnt[nacnt['dtype']=='object']['feat'].values)
```

    ['D_64' 'customer_ID' 'D_63' 'S_2']
    


```python
%%time
for c in nacnt[nacnt['dtype']=='object']['feat'].values:
    df = dd.read_parquet(train_file, columns=[c], engine='pyarrow')
    print(c, '-'*100)
    display(df.head(5))
    
    if c=='customer_ID':
        t = df[c].value_counts().compute()
        display(t.value_counts())
        display(df[df['customer_ID']=='8034aa3a67acb152f472bd8036f4c579b559d046ba12d7a911d27abd1c4b080b'].head(20))
        
    del df
    gc.collect()
```

    D_64 ----------------------------------------------------------------------------------------------------
    


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
      <th>D_64</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>O</td>
    </tr>
    <tr>
      <th>1</th>
      <td>O</td>
    </tr>
    <tr>
      <th>2</th>
      <td>O</td>
    </tr>
    <tr>
      <th>3</th>
      <td>O</td>
    </tr>
    <tr>
      <th>4</th>
      <td>O</td>
    </tr>
  </tbody>
</table>
</div>


    customer_ID ----------------------------------------------------------------------------------------------------
    


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
      <th>customer_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2e3fc76a68eeae6697b0a6b10f9288b7acaa678bbf99cd...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2e3fc76a68eeae6697b0a6b10f9288b7acaa678bbf99cd...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2e3fc76a68eeae6697b0a6b10f9288b7acaa678bbf99cd...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2e3fc76a68eeae6697b0a6b10f9288b7acaa678bbf99cd...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2e3fc76a68eeae6697b0a6b10f9288b7acaa678bbf99cd...</td>
    </tr>
  </tbody>
</table>
</div>



    13    386034
    12     10623
    10      6721
    9       6411
    8       6110
    2       6098
    11      5961
    3       5778
    6       5515
    7       5198
    1       5120
    4       4673
    5       4671
    Name: customer_ID, dtype: int64



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
      <th>customer_ID</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    D_63 ----------------------------------------------------------------------------------------------------
    


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
      <th>D_63</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CR</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CR</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CR</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CR</td>
    </tr>
  </tbody>
</table>
</div>


    S_2 ----------------------------------------------------------------------------------------------------
    


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
      <th>S_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-04-28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-05-25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-06-23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-07-29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-08-20</td>
    </tr>
  </tbody>
</table>
</div>


    CPU times: user 9 s, sys: 3.34 s, total: 12.3 s
    Wall time: 12.9 s
    

### int features

- only one feature is int type: `B_31`
- this feature should be treated as categorical feature as well


```python
print(nacnt[nacnt['dtype']=='int32']['feat'].values)
```

    ['B_31']
    


```python
%%time
for c in nacnt[nacnt['dtype']=='int32']['feat'].values:
    df = dd.read_parquet(train_file, columns=[c], engine='pyarrow')
    display(df[c].value_counts().compute().sort_index())
    
    del df
    gc.collect()
```


    0      16907
    1    5514544
    Name: B_31, dtype: int64


    CPU times: user 502 ms, sys: 52.3 ms, total: 555 ms
    Wall time: 561 ms
    

### float features


```python
nacnt.head(2)
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
      <th>feat</th>
      <th>dtype</th>
      <th>na_cnt</th>
      <th>nunique</th>
      <th>na_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>108</th>
      <td>D_87</td>
      <td>float32</td>
      <td>5527586</td>
      <td>1</td>
      <td>99.930127</td>
    </tr>
    <tr>
      <th>111</th>
      <td>D_88</td>
      <td>float32</td>
      <td>5525447</td>
      <td>6004</td>
      <td>99.891457</td>
    </tr>
  </tbody>
</table>
</div>




```python
nacnt[(nacnt['dtype']=='float32') & (nacnt['nunique']>100) & (nacnt['na_pct']<20)]
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
      <th>feat</th>
      <th>dtype</th>
      <th>na_cnt</th>
      <th>nunique</th>
      <th>na_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37</th>
      <td>S_7</td>
      <td>float32</td>
      <td>1020544</td>
      <td>4185390</td>
      <td>18.449843</td>
    </tr>
    <tr>
      <th>7</th>
      <td>S_3</td>
      <td>float32</td>
      <td>1020544</td>
      <td>4004312</td>
      <td>18.449843</td>
    </tr>
    <tr>
      <th>52</th>
      <td>D_62</td>
      <td>float32</td>
      <td>758161</td>
      <td>4519358</td>
      <td>13.706367</td>
    </tr>
    <tr>
      <th>19</th>
      <td>D_48</td>
      <td>float32</td>
      <td>718725</td>
      <td>4534865</td>
      <td>12.993426</td>
    </tr>
    <tr>
      <th>49</th>
      <td>D_61</td>
      <td>float32</td>
      <td>598052</td>
      <td>4529133</td>
      <td>10.811847</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>79</th>
      <td>B_24</td>
      <td>float32</td>
      <td>0</td>
      <td>4960342</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>77</th>
      <td>D_75</td>
      <td>float32</td>
      <td>0</td>
      <td>4446618</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75</th>
      <td>P_4</td>
      <td>float32</td>
      <td>0</td>
      <td>4702565</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>73</th>
      <td>B_23</td>
      <td>float32</td>
      <td>0</td>
      <td>5300841</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>95</th>
      <td>S_17</td>
      <td>float32</td>
      <td>0</td>
      <td>4996175</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>143 rows × 5 columns</p>
</div>




```python
print(cat_feats)
cat_feats = cat_feats + ['B_31']
print(cat_feats)
```

    ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68', 'B_31']
    


```python
float_feats = nacnt[nacnt['dtype']=='float32']['feat'].values.tolist()
print(len(float_feats))
float_feats = list((set(float_feats) - set(cat_feats)) - set(missing20) )
print(len(float_feats))
print(float_feats)
```

    185
    143
    ['R_17', 'B_40', 'R_27', 'S_18', 'B_13', 'B_33', 'R_20', 'S_6', 'R_23', 'R_16', 'B_24', 'D_125', 'D_44', 'D_91', 'D_71', 'P_2', 'B_15', 'D_103', 'S_12', 'D_144', 'D_123', 'D_94', 'D_70', 'D_39', 'P_4', 'S_23', 'R_12', 'S_5', 'D_72', 'B_27', 'B_6', 'D_89', 'D_143', 'D_80', 'B_3', 'B_28', 'R_11', 'B_14', 'B_1', 'D_124', 'D_109', 'B_25', 'B_36', 'B_5', 'B_18', 'D_61', 'R_13', 'B_37', 'S_7', 'D_104', 'B_26', 'B_4', 'R_6', 'D_133', 'B_21', 'S_19', 'D_115', 'R_18', 'D_45', 'D_69', 'S_24', 'D_84', 'S_17', 'B_12', 'D_52', 'R_24', 'D_127', 'R_14', 'D_113', 'D_83', 'D_141', 'B_10', 'S_22', 'D_96', 'R_15', 'S_25', 'D_54', 'D_60', 'D_59', 'S_11', 'R_8', 'D_74', 'R_4', 'D_118', 'D_62', 'B_7', 'S_15', 'B_2', 'R_28', 'S_26', 'D_119', 'D_86', 'D_81', 'D_93', 'R_3', 'B_16', 'B_9', 'D_107', 'D_78', 'D_140', 'S_13', 'B_11', 'D_47', 'R_1', 'D_55', 'R_22', 'D_102', 'D_112', 'D_131', 'B_32', 'R_10', 'R_7', 'R_19', 'D_41', 'D_130', 'B_23', 'R_5', 'D_121', 'B_19', 'P_3', 'B_8', 'D_79', 'D_122', 'S_3', 'R_25', 'D_92', 'D_58', 'D_51', 'B_41', 'S_8', 'B_22', 'D_139', 'R_2', 'D_48', 'D_145', 'D_129', 'B_20', 'S_16', 'S_20', 'D_128', 'D_75', 'D_65', 'R_21']
    


```python
%%time
#features with negative values will not have the log transform plot
import matplotlib.pyplot as plt
import math

float_stats = []


ncols = 4
nrows = math.ceil(len(float_feats)*2/ncols)
# nrows = math.ceil(50*2/ncols)
# ncols = 4
# nrows = math.ceil(10*2/ncols)
#fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 360))

for i, c in enumerate(float_feats):
    df = dd.read_parquet(train_file, columns=[c], engine='pyarrow')
    
    #--calculate the stats ----------
    item = [c, df[c].compute().min(), df[c].compute().max(), df[c].compute().mean(), df[c].compute().std(), df[c].compute().median(), df[c].compute().skew()]
    for q in [0.05, 0.15, 0.25, 0.75, 0.85, 0.95]:
        item.append(df[c].compute().quantile(q=q))

    float_stats.append(item)
    
    #--plot the graph------------------
    df[c].compute().plot(ax=axes[i//2, (i%2)*2], kind='hist', bins=50)
    axes[i//2, (i%2)*2].set_title(f'{c}')
    
    if df.compute().min().values[0]>0:
        df[c].compute().transform(np.log).plot(ax=axes[i//2,(i%2)*2 + 1], kind='hist', bins=50)
        axes[i//2,(i%2)*2 + 1].set_title(f'log of {c}')

    del df
    gc.collect()

```

    CPU times: user 15min 51s, sys: 1min 56s, total: 17min 47s
    Wall time: 17min 41s
    


    
![png](amex-default-prediction-eda-of-train-data/output_45_1.png)
    



```python
float_stats_df = pd.DataFrame(data=float_stats, 
                              columns=['feat', 'min', 'max', 'mean', 'std', 'median', 'skew', 
                                       'quantile0.05', 'quantile0.15', 'quantile0.25', 'quantile0.75', 'quantile0.85', 'quantile0.95'])
```


```python
float_stats_df
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
      <th>feat</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>std</th>
      <th>median</th>
      <th>skew</th>
      <th>quantile0.05</th>
      <th>quantile0.15</th>
      <th>quantile0.25</th>
      <th>quantile0.75</th>
      <th>quantile0.85</th>
      <th>quantile0.95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>R_17</td>
      <td>1.066114e-09</td>
      <td>1.001528</td>
      <td>0.005319</td>
      <td>0.006380</td>
      <td>0.005031</td>
      <td>26.054792</td>
      <td>0.000503</td>
      <td>0.001511</td>
      <td>0.002519</td>
      <td>0.007544</td>
      <td>0.008551</td>
      <td>0.009559</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B_40</td>
      <td>1.201275e-08</td>
      <td>5755.076172</td>
      <td>0.203261</td>
      <td>8.081345</td>
      <td>0.058309</td>
      <td>475.881531</td>
      <td>0.003281</td>
      <td>0.008622</td>
      <td>0.017393</td>
      <td>0.245154</td>
      <td>0.402013</td>
      <td>0.630622</td>
    </tr>
    <tr>
      <th>2</th>
      <td>R_27</td>
      <td>-2.571099e-02</td>
      <td>1.010000</td>
      <td>0.893695</td>
      <td>0.312307</td>
      <td>1.004365</td>
      <td>-2.449320</td>
      <td>0.015879</td>
      <td>1.000420</td>
      <td>1.001545</td>
      <td>1.007182</td>
      <td>1.008310</td>
      <td>1.009437</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S_18</td>
      <td>4.075928e-11</td>
      <td>1.010000</td>
      <td>0.031466</td>
      <td>0.160546</td>
      <td>0.005134</td>
      <td>5.897257</td>
      <td>0.000512</td>
      <td>0.001542</td>
      <td>0.002571</td>
      <td>0.007701</td>
      <td>0.008730</td>
      <td>0.009757</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B_13</td>
      <td>1.905850e-08</td>
      <td>276.177826</td>
      <td>0.100716</td>
      <td>0.559384</td>
      <td>0.029314</td>
      <td>177.815186</td>
      <td>0.002531</td>
      <td>0.006215</td>
      <td>0.009255</td>
      <td>0.089415</td>
      <td>0.151466</td>
      <td>0.349563</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>138</th>
      <td>S_20</td>
      <td>6.887991e-10</td>
      <td>1.010000</td>
      <td>0.017487</td>
      <td>0.111088</td>
      <td>0.005062</td>
      <td>8.771165</td>
      <td>0.000506</td>
      <td>0.001521</td>
      <td>0.002533</td>
      <td>0.007592</td>
      <td>0.008605</td>
      <td>0.009619</td>
    </tr>
    <tr>
      <th>139</th>
      <td>D_128</td>
      <td>3.235349e-09</td>
      <td>1.023472</td>
      <td>0.584121</td>
      <td>0.493150</td>
      <td>1.000419</td>
      <td>-0.322798</td>
      <td>0.001188</td>
      <td>0.003570</td>
      <td>0.005949</td>
      <td>1.004735</td>
      <td>1.006463</td>
      <td>1.008185</td>
    </tr>
    <tr>
      <th>140</th>
      <td>D_75</td>
      <td>1.120811e-09</td>
      <td>4.275118</td>
      <td>0.171199</td>
      <td>0.224214</td>
      <td>0.074370</td>
      <td>2.284789</td>
      <td>0.001344</td>
      <td>0.004032</td>
      <td>0.006720</td>
      <td>0.268856</td>
      <td>0.342966</td>
      <td>0.607340</td>
    </tr>
    <tr>
      <th>141</th>
      <td>D_65</td>
      <td>3.539404e-09</td>
      <td>385.901642</td>
      <td>0.039764</td>
      <td>0.473098</td>
      <td>0.005223</td>
      <td>259.727264</td>
      <td>0.000521</td>
      <td>0.001568</td>
      <td>0.002611</td>
      <td>0.007836</td>
      <td>0.008881</td>
      <td>0.009927</td>
    </tr>
    <tr>
      <th>142</th>
      <td>R_21</td>
      <td>3.646872e-09</td>
      <td>1.010000</td>
      <td>0.021610</td>
      <td>0.127847</td>
      <td>0.005081</td>
      <td>7.558036</td>
      <td>0.000509</td>
      <td>0.001523</td>
      <td>0.002537</td>
      <td>0.007624</td>
      <td>0.008642</td>
      <td>0.009661</td>
    </tr>
  </tbody>
</table>
<p>143 rows × 13 columns</p>
</div>




```python
float_stats_df.to_csv('float_stats.csv', sep='|', index=False)
```


```python

```
