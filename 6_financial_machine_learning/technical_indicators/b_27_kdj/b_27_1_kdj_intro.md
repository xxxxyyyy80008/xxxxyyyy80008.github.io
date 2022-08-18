## KDJ

**References**

- [futunn: KDJ](https://www.futunn.com/en/learn/detail-what-is-kdj-64858-0)


**Definition**

- KDJ, also known as random index, is a technical index widely used in short-term trend analysis of futures and stock markets.
- KDJ is calculated on the basis of the highest price, the lowest price and the closing price. It can reflect the intensity of price fluctuations, overbought and oversold, and give trading signals before prices rise or fall. 
- KDJ is sensitive to price changes, which may generate wrong trading signals in very volatile markets, causing prices not to rise or fall with the signals, thus causing traders to make misjudgments.

**Calculation**

---


step 1: calculate the immature random value (RSV)

![png](img/kdj1.png)

-  *Hn denotes the highest price, Ln denotes the lowest price, C denotes the closing price*

step 2: calculate the %K line:

![png](img/kdj2.png)

step 3: calculate the %D line:


![png](img/kdj3.png)

step 4: calculate the %J line:

![png](img/kdj4.png)

---

**Read the indicator**


- KDJ values range from 0 to 100 (J values sometimes exceed). Generally speaking, an overbought signal occurs when the D value is more than 70 and an oversell signal occurs when the D value is less than 30.

- Gold fork
    - When the K line breaks through the D line on the graph, it is commonly known as the golden fork, which is a buy signal. In addition, when the K-line and D-line cross upward below 20, the short-term buy signal is more accurate; if the K value is below 50 and crosses twice above D value to form a higher golden fork "W" shape, then the stock price may rise considerably and the market prospect is promising.

- Dead fork
    - When the K value gets smaller and smaller, and then falls below the D line from above, it is often called a dead fork and is regarded as a sell signal. In addition, when K-line and D-line cross downward at gate 80, the short-term sell signal is more accurate. If the K value is above 50, crossing below the D line twice in the trend, and from the low dead cross "M" shape, the market outlook may have a considerable decline in stock prices.

- Bottom and top
    - J-line is a sensitive line of direction. When the J value is greater than 90, especially for more than 5 consecutive days, the stock price will form at least a short-term peak. On the contrary, when the J value is less than 10:00, especially for several consecutive days, the stock price will form at least a short-term bottom.


##### Load basic packages 


```python
import pandas as pd
import numpy as np
import os
import gc
import copy
from pathlib import Path
from datetime import datetime, timedelta, time, date
```


```python
#this package is to download equity price data from yahoo finance
#the source code of this package can be found here: https://github.com/ranaroussi/yfinance/blob/main
import yfinance as yf
```


```python
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
random_seed=1234
pl.seed_everything(random_seed)
```

    Global seed set to 1234
    




    1234




```python
#S&P 500 (^GSPC),  Dow Jones Industrial Average (^DJI), NASDAQ Composite (^IXIC)
#Russell 2000 (^RUT), Crude Oil Nov 21 (CL=F), Gold Dec 21 (GC=F)
#Treasury Yield 10 Years (^TNX)

#benchmark_tickers = ['^GSPC', '^DJI', '^IXIC', '^RUT',  'CL=F', 'GC=F', '^TNX']

benchmark_tickers = ['^GSPC']
```


```python
#https://github.com/ranaroussi/yfinance/blob/main/yfinance/base.py
#     def history(self, period="1mo", interval="1d",
#                 start=None, end=None, prepost=False, actions=True,
#                 auto_adjust=True, back_adjust=False,
#                 proxy=None, rounding=False, tz=None, timeout=None, **kwargs):

dfs = {}

for ticker in benchmark_tickers:
    cur_data = yf.Ticker(ticker)
    hist = cur_data.history(period="max", start='2000-01-01')
    print(datetime.now(), ticker, hist.shape, hist.index.min(), hist.index.max())
    dfs[ticker] = hist
```

    2022-08-18 18:33:51.914152 ^GSPC (5694, 7) 1999-12-31 00:00:00 2022-08-17 00:00:00
    


```python
ticker = '^GSPC'
dfs[ticker].tail(5)
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
      <th>2022-08-11</th>
      <td>4227.399902</td>
      <td>4257.910156</td>
      <td>4201.410156</td>
      <td>4207.270020</td>
      <td>3925060000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-08-12</th>
      <td>4225.020020</td>
      <td>4280.470215</td>
      <td>4219.779785</td>
      <td>4280.149902</td>
      <td>3252290000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-08-15</th>
      <td>4269.370117</td>
      <td>4301.790039</td>
      <td>4256.899902</td>
      <td>4297.140137</td>
      <td>3087740000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-08-16</th>
      <td>4290.459961</td>
      <td>4325.279785</td>
      <td>4277.770020</td>
      <td>4305.200195</td>
      <td>3792010000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-08-17</th>
      <td>4280.399902</td>
      <td>4302.180176</td>
      <td>4253.080078</td>
      <td>4274.040039</td>
      <td>3293430000</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



##### Define KDJ calculation function




```python
def cal_kdj(ohlc: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    KDJ
    reference:  https://www.futunn.com/en/learn/detail-what-is-kdj-64858-0

    """
    ohlc = ohlc.copy(deep=True)
    ohlc.columns = [c.lower() for c in ohlc.columns]

    highest_high = ohlc["high"].rolling(center=False, window=period).max()
    lowest_low = ohlc["low"].rolling(center=False, window=period).min()
    rsv = (ohlc["close"] - lowest_low) / (highest_high - lowest_low) * 100
    rsv = rsv.values

    k_ = np.zeros(len(ohlc))
    d_ = np.zeros(len(ohlc))

    for i in range(len(ohlc)):
        if i < period:
            k_[i] = 0
            d_[i] = 0
        else:
            k_[i] = (2/3)*k_[i-1] + (1/3)*rsv[i]
            d_[i] = (2/3)*d_[i-1] + (1/3)*k_[i]
    j_ = 3*k_ - 2*d_


    return pd.DataFrame(data={'K': k_, 'D': d_, 'J': j_}, index=ohlc.index)

```

##### Calculate KDJ


```python
df = dfs[ticker][['Open', 'High', 'Low', 'Close', 'Volume']]
```


```python
df = df.round(2)
```


```python
cal_kdj
```




    <function __main__.cal_kdj(ohlc: pandas.core.frame.DataFrame, period: int = 14) -> pandas.core.frame.DataFrame>




```python
df_ta = cal_kdj(df, period = 14)
df = df.merge(df_ta, left_index = True, right_index = True, how='inner' )

del df_ta
gc.collect()
```




    106




```python
display(df.head(5))
display(df.tail(5))
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
      <th>K</th>
      <th>D</th>
      <th>J</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1999-12-31</th>
      <td>1464.47</td>
      <td>1472.42</td>
      <td>1458.19</td>
      <td>1469.25</td>
      <td>374050000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2000-01-03</th>
      <td>1469.25</td>
      <td>1478.00</td>
      <td>1438.36</td>
      <td>1455.22</td>
      <td>931800000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>1455.22</td>
      <td>1455.22</td>
      <td>1397.43</td>
      <td>1399.42</td>
      <td>1009000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>1399.42</td>
      <td>1413.27</td>
      <td>1377.68</td>
      <td>1402.11</td>
      <td>1085500000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>1402.11</td>
      <td>1411.90</td>
      <td>1392.10</td>
      <td>1403.45</td>
      <td>1092300000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>K</th>
      <th>D</th>
      <th>J</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-08-11</th>
      <td>4227.40</td>
      <td>4257.91</td>
      <td>4201.41</td>
      <td>4207.27</td>
      <td>3925060000</td>
      <td>88.639826</td>
      <td>89.312614</td>
      <td>87.294250</td>
    </tr>
    <tr>
      <th>2022-08-12</th>
      <td>4225.02</td>
      <td>4280.47</td>
      <td>4219.78</td>
      <td>4280.15</td>
      <td>3252290000</td>
      <td>92.397701</td>
      <td>90.340976</td>
      <td>96.511150</td>
    </tr>
    <tr>
      <th>2022-08-15</th>
      <td>4269.37</td>
      <td>4301.79</td>
      <td>4256.90</td>
      <td>4297.14</td>
      <td>3087740000</td>
      <td>94.489398</td>
      <td>91.723783</td>
      <td>100.020628</td>
    </tr>
    <tr>
      <th>2022-08-16</th>
      <td>4290.46</td>
      <td>4325.28</td>
      <td>4277.77</td>
      <td>4305.20</td>
      <td>3792010000</td>
      <td>94.312082</td>
      <td>92.586550</td>
      <td>97.763147</td>
    </tr>
    <tr>
      <th>2022-08-17</th>
      <td>4280.40</td>
      <td>4302.18</td>
      <td>4253.08</td>
      <td>4274.04</td>
      <td>3293430000</td>
      <td>89.266658</td>
      <td>91.479919</td>
      <td>84.840136</td>
    </tr>
  </tbody>
</table>
</div>



```python
df[['K', 'D', 'J']].hist(bins=50)
```




    array([[<AxesSubplot:title={'center':'K'}>,
            <AxesSubplot:title={'center':'D'}>],
           [<AxesSubplot:title={'center':'J'}>, <AxesSubplot:>]], dtype=object)




    
![png](img/output_16_1.png)
    



```python
#https://github.com/matplotlib/mplfinance
#this package help visualize financial data
import mplfinance as mpf
import matplotlib.colors as mcolors

# all_colors = list(mcolors.CSS4_COLORS.keys())#"CSS Colors"
all_colors = list(mcolors.TABLEAU_COLORS.keys()) # "Tableau Palette",
# all_colors = list(mcolors.BASE_COLORS.keys()) #"Base Colors",


#https://github.com/matplotlib/mplfinance/issues/181#issuecomment-667252575
#list of colors: https://matplotlib.org/stable/gallery/color/named_colors.html
#https://github.com/matplotlib/mplfinance/blob/master/examples/styles.ipynb

def make_3panels2(main_data, mid_panel, chart_type='candle', names=None, 
                  figratio=(14,9), fill_weights = (0, 0)):
    """
    main chart type: default is candle. alternatives: ohlc, line

    example:
    start = 200

    names = {'main_title': 'MAMA: MESA Adaptive Moving Average', 
             'sub_tile': 'S&P 500 (^GSPC)', 'y_tiles': ['price', 'Volume [$10^{6}$]']}


    make_candle(df.iloc[-start:, :5], df.iloc[-start:][['MAMA', 'FAMA']], names = names)
    
    """

    style = mpf.make_mpf_style(base_mpf_style='yahoo',  #charles
                               base_mpl_style = 'seaborn-whitegrid',
#                                marketcolors=mpf.make_marketcolors(up="r", down="#0000CC",inherit=True),
                               gridcolor="whitesmoke", 
                               gridstyle="--", #or None, or - for solid
                               gridaxis="both", 
                               edgecolor = 'whitesmoke',
                               facecolor = 'white', #background color within the graph edge
                               figcolor = 'white', #background color outside of the graph edge
                               y_on_right = False,
                               rc =  {'legend.fontsize': 'small',#or number
                                      #'figure.figsize': (14, 9),
                                     'axes.labelsize': 'small',
                                     'axes.titlesize':'small',
                                     'xtick.labelsize':'small',#'x-small', 'small','medium','large'
                                     'ytick.labelsize':'small'
                                     }, 
                              )   

    if (chart_type is None) or (chart_type not in ['ohlc', 'line', 'candle', 'hollow_and_filled']):
        chart_type = 'candle'
    len_dict = {'candle':2, 'ohlc':3, 'line':1, 'hollow_and_filled':2}    
        
    kwargs = dict(type=chart_type, figratio=figratio, volume=True, volume_panel=2, 
                  panel_ratios=(4,2,1), tight_layout=True, style=style, returnfig=True)
    
    if names is None:
        names = {'main_title': '', 'sub_tile': ''}
    


    added_plots = { }

    
    i = 0
    for name_, data_ in mid_panel.iteritems():
        added_plots[name_] = mpf.make_addplot(data_, panel=1, color=all_colors[i], secondary_y=False)
        i = i + 1
    

    fig, axes = mpf.plot(main_data,  **kwargs,
                         addplot=list(added_plots.values()),
                        )
    # add a new suptitle
    fig.suptitle(names['main_title'], y=1.05, fontsize=12, x=0.128)

    axes[0].set_title(names['sub_tile'], fontsize=10, style='italic',  loc='left')
    axes[2].set_ylabel('KDJ')

#     axes[0].set_ylabel(names['y_tiles'][0])
#     axes[2].set_ylabel(names['y_tiles'][1])
    return fig, axes
   
```


```python

start = -200
end = -100#df.shape[0]

names = {'main_title': 'KDJ', 
         'sub_tile': f'{ticker}: an overbought signal occurs when the D value is more than 70 and an oversell signal occurs when the D value is less than 30'}


aa_, bb_ = make_3panels2(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
             df.iloc[start:end][['K', 'D', 'J']], 
             chart_type='hollow_and_filled',names = names, 
                         fill_weights = (-30, 30))
```


    
![png](img/output_18_0.png)
    

