---
layout: default
title: Chaikin Oscillator
parent: Technical Indicators
nav_order: 11
---

## Chaikin Oscillator

**References**

- [tradingview: Chaikin Oscillator](https://www.tradingview.com/support/solutions/43000501979-chaikin-oscillator/)


**Definition**

- The Chaikin Oscillator is an **indicator of an indicator** created by *Marc Chaikin*. 
- The Chaikin Oscillator takes Accumulation/Distribution Line (ADL) and applies two Exponential Moving Averages of varying length to the line. 
- The Chaikin Oscillator's value is then derived by subtracting the longer term EMA of the ADL from the shorter term EMA of the ADL. 
- it measures the momentum of the ADL by plotting a line which fluctuates between positive and negative values. 
    - changes in momentum often precede changes in trend.


**Calculation**

---

There are four steps in calculating The Chaikin Oscillator. the following example is for a (3,10) Period Chaikin Oscillator:


1. Find the Money Flow Multiplier

    `Money Flow Multiplier = [(Close  -  Low) - (High - Close)] /(High - Low)`

2. Calculate Money Flow Volume

    `Money Flow Volume = Money Flow Multiplier x Volume for the Period`

3. Determine ADL

    `ADL = Previous ADL + Current Period Money Flow Volume`

4. Apply EMA (user defined periods) to the ADL to generate the Chaikin Oscillator

    `Chaikin Oscillator = (3-day EMA of ADL)  -  (10-day EMA of ADL)`

---



**Read the indicator**

- The Chaikin Oscillator calculates a value that fluctuates between positive and negative values.
    - When The Chaikin Oscillator's value is above 0, **buying pressure is higher.**
    - When The Chaikin Oscillator's value is below 0, **selling pressure is higher.**

- **Crosses**
    - When The Chaikin Oscillator crosses the Zero Line, this can be an indication that there is an impending trend reversal.
    - Bullish Crosses occur when The Chaikin Oscillator crosses from Below the Zero Line to Above the Zero Line. Price then rises.
    - Bearish Crosses occur when The Chaikin Oscillator crosses from Above the Zero Line to Below the Zero Line. Price then falls.

- **Divergence**
    - Chaikin Oscillator Divergence occurs when there is a difference between what price action is indicating and what The Chaikin Oscillator is indicating. These differences can be interpreted as an impending reversal.
    - Bullish Chaikin Oscillator Divergence is when price makes a new low but the Chaikin Oscillator makes a higher low.
    - Bearish Chaikin Oscillator Divergence is when price makes a new high but the Chaikin Oscillator makes a lower high.


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

    2022-08-25 22:27:54.523101 ^GSPC (5700, 7) 1999-12-31 00:00:00 2022-08-25 00:00:00
    


```python
dfs['^GSPC'].tail(5)
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
      <th>2022-08-19</th>
      <td>4266.310059</td>
      <td>4266.310059</td>
      <td>4218.700195</td>
      <td>4228.479980</td>
      <td>3210680000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-08-22</th>
      <td>4195.080078</td>
      <td>4195.080078</td>
      <td>4129.859863</td>
      <td>4137.990234</td>
      <td>3365220000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-08-23</th>
      <td>4133.089844</td>
      <td>4159.770020</td>
      <td>4124.029785</td>
      <td>4128.729980</td>
      <td>3117800000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-08-24</th>
      <td>4126.549805</td>
      <td>4156.560059</td>
      <td>4119.970215</td>
      <td>4140.770020</td>
      <td>3056910000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-08-25</th>
      <td>4153.259766</td>
      <td>4180.910156</td>
      <td>4147.589844</td>
      <td>4179.109863</td>
      <td>396298432</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



##### Define Chaikin Oscillator calculation function




```python

#https://github.com/peerchemist/finta/blob/af01fa594995de78f5ada5c336e61cd87c46b151/finta/finta.py


def cal_adl(ohlc: pd.DataFrame) -> pd.Series:
    """
    The accumulation/distribution line was created by Marc Chaikin to determine the flow of money into or out of a security.
    
    It should not be confused with the advance/decline line. While their initials might be the same, 
    these are entirely different indicators, and their uses are different as well. 
    
    Whereas the advance/decline line can provide insight into market movements, the accumulation/distribution line 
    is of use to traders looking to measure buy/sell pressure on a security or confirm the strength of a trend.
    
    """
    ohlc = ohlc.copy()
    ohlc.columns = [c.lower() for c in ohlc.columns]
    
    MFM = pd.Series(
        ((ohlc["close"] - ohlc["low"])
        - (ohlc["high"] - ohlc["close"])) / (ohlc["high"] - ohlc["low"]),
        name="MFM",
    )  # Money flow multiplier
    
    MFV = pd.Series(MFM * ohlc["volume"], name="MFV")
    
    return MFV.cumsum()


def cal_chaikin(ohlc: pd.DataFrame, slow_period: int = 10, fast_period: int = 3, adjust: bool = True) -> pd.Series:
    """
     Chaikin Oscillator, named after its creator, Marc Chaikin, the Chaikin oscillator is an oscillator that measures the accumulation/distribution
     line of the moving average convergence divergence (MACD). The Chaikin oscillator is calculated by subtracting a 10-day exponential moving average (EMA)
     of the accumulation/distribution line from a three-day EMA of the accumulation/distribution line, and highlights the momentum implied by the
     accumulation/distribution line.
     
     1. Find the Money Flow Multiplier
        Money Flow Multiplier = [(Close  -  Low) - (High - Close)] /(High - Low)
     2. Calculate Money Flow Volume
        Money Flow Volume = Money Flow Multiplier x Volume for the Period
     3. Determine ADL
        ADL = Previous ADL + Current Period Money Flow Volume
     4. Apply EMA (user defined periods) to the ADL to generate the Chaikin Oscillator
        Chaikin Oscillator = (3-day EMA of ADL)  -  (10-day EMA of ADL)    
     
     
     """
    
    adl = cal_adl(ohlc)
    fast_ewm = adl.ewm(span=fast_period, min_periods=max(fast_period-1, 0), adjust=adjust).mean()
    slow_ewm = adl.ewm(span=slow_period, min_periods=max(slow_period-1, 0), adjust=adjust).mean()

    return pd.Series(fast_ewm - slow_ewm, name=f'CHAIKIN')
```

##### Calculate Chaikin Oscillator


```python
df = dfs['^GSPC'][['Open', 'High', 'Low', 'Close', 'Volume']]
```


```python
df = df.round(2)
```


```python
cal_chaikin
```




    <function __main__.cal_chaikin(ohlc: pandas.core.frame.DataFrame, slow_period: int = 10, fast_period: int = 3, adjust: bool = True) -> pandas.core.series.Series>




```python
df_ta = cal_chaikin(df, slow_period=10, fast_period=3)
df = df.merge(df_ta, left_index = True, right_index = True, how='inner' )

del df_ta
gc.collect()
```




    122




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
      <th>CHAIKIN</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-03</th>
      <td>1469.25</td>
      <td>1478.00</td>
      <td>1438.36</td>
      <td>1455.22</td>
      <td>931800000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>1455.22</td>
      <td>1455.22</td>
      <td>1397.43</td>
      <td>1399.42</td>
      <td>1009000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>1399.42</td>
      <td>1413.27</td>
      <td>1377.68</td>
      <td>1402.11</td>
      <td>1085500000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>1402.11</td>
      <td>1411.90</td>
      <td>1392.10</td>
      <td>1403.45</td>
      <td>1092300000</td>
      <td>NaN</td>
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
      <th>CHAIKIN</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>2022-08-19</th>
      <td>4266.31</td>
      <td>4266.31</td>
      <td>4218.70</td>
      <td>4228.48</td>
      <td>3210680000</td>
      <td>1.996214e+09</td>
    </tr>
    <tr>
      <th>2022-08-22</th>
      <td>4195.08</td>
      <td>4195.08</td>
      <td>4129.86</td>
      <td>4137.99</td>
      <td>3365220000</td>
      <td>6.548373e+08</td>
    </tr>
    <tr>
      <th>2022-08-23</th>
      <td>4133.09</td>
      <td>4159.77</td>
      <td>4124.03</td>
      <td>4128.73</td>
      <td>3117800000</td>
      <td>-6.845520e+08</td>
    </tr>
    <tr>
      <th>2022-08-24</th>
      <td>4126.55</td>
      <td>4156.56</td>
      <td>4119.97</td>
      <td>4140.77</td>
      <td>3056910000</td>
      <td>-1.037074e+09</td>
    </tr>
    <tr>
      <th>2022-08-25</th>
      <td>4153.26</td>
      <td>4180.91</td>
      <td>4147.59</td>
      <td>4179.11</td>
      <td>396298432</td>
      <td>-9.745365e+08</td>
    </tr>
  </tbody>
</table>
</div>



```python
df['CHAIKIN'].hist(bins=50)
```




    <AxesSubplot:>




    
![png](output_16_1.png)
    



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
  
    fb_bbands2_ = dict(y1=fill_weights[0]*np.ones(mid_panel.shape[0]),
                      y2=fill_weights[1]*np.ones(mid_panel.shape[0]),color="lightskyblue",alpha=0.1,interpolate=True)
    fb_bbands2_['panel'] = 1

    fb_bbands= [fb_bbands2_]
    
    
    i = 0
    for name_, data_ in mid_panel.iteritems():
        added_plots[name_] = mpf.make_addplot(data_, panel=1, color=all_colors[i])
        i = i + 1
    

    fig, axes = mpf.plot(main_data,  **kwargs,
                         addplot=list(added_plots.values()), 
                         fill_between=fb_bbands)
    # add a new suptitle
    fig.suptitle(names['main_title'], y=1.05, fontsize=12, x=0.1285)

    axes[0].set_title(names['sub_tile'], fontsize=10, style='italic',  loc='left')
    axes[2].set_ylabel('CHAIKIN')

#     axes[0].set_ylabel(names['y_tiles'][0])
#     axes[2].set_ylabel(names['y_tiles'][1])
    return fig, axes
   
```


```python

start = -400
end = -300

names = {'main_title': f'{ticker}', 
         'sub_tile': 'Chaikin Oscillator indicator: values crosses from Below the Zero Line to Above the Zero Line, price may then rise;'}


aa_, bb_ = make_3panels2(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
             df.iloc[start:end][['CHAIKIN']], 
             chart_type='hollow_and_filled',names = names, 
                         fill_weights = (-2, 2))
```


    
![png](output_18_0.png)
    

