---
title: How to process a large csv file with pandas and parquet
---

### How to process a large csv file with pandas and parquet

*Jul 8, 2022*

Due to constraints of RAM, large csv files typically can not be loaded to RAM at once. For example, it will for sure trigger memory error when trying to load a 19GB file to 16GB RAM. 

Pandas `read_csv` function provides the flexibility to read a large csv file in chunks by setting the chunksize to an appropriate number.  
Parquet is a column-oriented data storage format in the Apache Hadoop ecosystem.  Data stored in parquet format allows data analysis and manipulation by column in high speed. 

The following provides a quick guide on how to use pandas and parquet to read and write large datasets. The file size of the original csv is about 16.3GB, and the output parquet file is about 3GB. 

#### note:

- original file is my *public notebook* on Kaggle: https://www.kaggle.com/code/xxxxyyyy80008/process-amex-train-data-to-parquet-format



#### references:

1. read file by chunksize
 - https://stackoverflow.com/questions/25962114/how-do-i-read-a-large-csv-file-with-pandas
 - https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
1. convert 64bit numeric values to 32bit values: convert int64 to int32; convert float64 to float32
 - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html
 - **warning**: if the values is out of 32bit value range, the conversion will be erroneous
1. save data to parquet file: use compression='GZIP' to further decrease file size
 - https://arrow.apache.org/docs/python/parquet.html
 - https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html?highlight=write_table

#### steps:

1. read csv by chunk. can set chunk size as 2 millon (i.e. chunksize = 20e5)
1. convert int64 to int32 and float64 to float32 >> this will cut the file size to half
1. save each chunk to a parquet file



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
```


```python
%%time
#https://stackoverflow.com/questions/25962114/how-do-i-read-a-large-csv-file-with-pandas
chunksize = 20e5
print('chunksize=', chunksize)

def process_big_csv(chunk, dest_file):
    #---convert float64 to float32--------
    float64_cols = chunk.select_dtypes(include=['float64']).columns.tolist()
    chunk[float64_cols] = np.float32(chunk[float64_cols].values)
    #---convert int64 to int32
    int64_cols = chunk.select_dtypes(include=['int64']).columns.tolist()
    chunk[int64_cols] = np.int32(chunk[int64_cols].values)
    
    #-- save to parquet file
    table = pa.Table.from_pandas(chunk)
    pq.write_table(table, dest_file, compression = 'GZIP')
    
    del table, chunk
    gc.collect()

train_file = '/kaggle/input/amex-default-prediction/train_data.csv'
with pd.read_csv(train_file, chunksize=chunksize) as reader:
    for i, chunk in enumerate(reader):
        dest_file = f'{i+1}.parquet'
        process_big_csv(chunk, dest_file)
```

    chunksize= 2000000.0
    CPU times: user 10min, sys: 3min 4s, total: 13min 5s
    Wall time: 15min 6s
    

## check output file


```python
%%time
train = pd.read_parquet('1.parquet')
```

    CPU times: user 14.5 s, sys: 3.03 s, total: 17.6 s
    Wall time: 9.98 s
    


```python
train.shape
```




    (2000000, 190)




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2000000 entries, 0 to 1999999
    Columns: 190 entries, customer_ID to D_145
    dtypes: float32(185), int32(1), object(4)
    memory usage: 1.4+ GB
    

## display files and names


```python
files = next(os.walk('.'))[2]
parquet_files = []
for file in files:
    if '.parquet' in file:
        parquet_files.append(file)

len(parquet_files), parquet_files[:2]
```




    (3, ['2.parquet', '1.parquet'])




```python
!ls -lh
```

    total 3.1G
    -rw-r--r-- 1 root root 1.1G Jul  7 15:10 1.parquet
    -rw-r--r-- 1 root root 1.1G Jul  7 15:15 2.parquet
    -rw-r--r-- 1 root root 870M Jul  7 15:19 3.parquet
    ---------- 1 root root  263 Jul  7 15:04 __notebook_source__.ipynb
    
