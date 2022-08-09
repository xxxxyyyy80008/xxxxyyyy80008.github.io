---
sort: 1
title: Bollinger Bands Introduction
---


## Bollinger Bands Introduction

**References:**

- [Bollinger Bands (BB)](https://www.tradingview.com/support/solutions/43000501971-bollinger-bands-b-b/)
- [Bollinger Bands %B](https://www.tradingview.com/support/solutions/43000501971-bollinger-bands-b-b/)
- [Bollinger Bands Width (BBW)](https://www.tradingview.com/support/solutions/43000501972-bollinger-bands-width-bbw/)


**Bollinger Bands (BB)**
 
    Definition
    Bollinger Bands (BB) were created by John Bollinger in the early 1980’s to fill the need to visualize changes in volatility. 
    Bollinger Bands consist of three lines:
        - the middle line is 20-day Simple Moving Average (SMA). (20 days can be replaced by other periods, such as 10 days)
        - the upper line is several - typically 2 - standard deviations above the middle line
        - the lower line is same number of standard deviations below the middle line
       
    Calculation
        Middle Band – 20 Day Simple Moving Average
        Upper Band – 20 Day Simple Moving Average + (Standard Deviation x 2)
        Lower Band – 20 Day Simple Moving Average - (Standard Deviation x 2)
        
        
    MOBO bands are based on a zone of 0.80 standard deviation with a 10 period look-back. If the price breaks out of the MOBO band it can signify a trend move or price spike contains 42% of price movements(noise) within bands.

**Bollinger Bands %B** or **Percent Bandwidth (%B)**

    Definition
    Bollinger Bands %B or Percent Bandwidth (%B) is an indicator derived from the standard Bollinger Bands (BB). 
    John Bollinger introduced %B in 2010.

    Calculation
        %B = (Current Price - Lower Band) / (Upper Band - Lower Band)
        
    %B Above 1 = Price is Above the Upper Band
    %B Equal to 1 = Price is at the Upper Band
    %B Above .50 = Price is Above the Middle Line
    %B Below .50 = Price is Below the Middle Line
    %B Equal to 0 = Price is at the Lower Band
    %B Below 0 = Price is Below the Lower Band
    
**Bollinger Bands Width (BBW)**    

    Bollinger Bands Width (BBW) is derived from the standard Bollinger Bands.
    John Bollinger introduced Bollinger Bands Width in 2010.

    Calculation
    Bollinger Bands Width = (Upper Band - Lower Band) / Middle Band

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

    2022-08-09 19:33:29.807667 ^GSPC (5687, 7) 1999-12-31 00:00:00 2022-08-08 00:00:00
    


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
      <th>2022-08-02</th>
      <td>4104.209961</td>
      <td>4140.470215</td>
      <td>4079.810059</td>
      <td>4091.189941</td>
      <td>3880790000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-08-03</th>
      <td>4107.959961</td>
      <td>4167.660156</td>
      <td>4107.959961</td>
      <td>4155.169922</td>
      <td>3544410000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-08-04</th>
      <td>4154.850098</td>
      <td>4161.290039</td>
      <td>4135.419922</td>
      <td>4151.939941</td>
      <td>3565810000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-08-05</th>
      <td>4115.870117</td>
      <td>4151.580078</td>
      <td>4107.310059</td>
      <td>4145.189941</td>
      <td>3540260000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-08-08</th>
      <td>4155.930176</td>
      <td>4186.620117</td>
      <td>4128.970215</td>
      <td>4140.060059</td>
      <td>3604650000</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



##### Define Bollinger Bands calculation function




```python
#https://github.com/peerchemist/finta/blob/af01fa594995de78f5ada5c336e61cd87c46b151/finta/finta.py#L935
def cal_bbands(ohlc: pd.DataFrame, period: int = 20, column: str = "close", 
               MA: pd.Series = None, std_multiplier: float = 2) -> pd.DataFrame:
    
    """
     Developed by John Bollinger, Bollinger Bands® are volatility bands placed above and below a moving average.
     Volatility is based on the standard deviation, which changes as volatility increases and decreases.
     The bands automatically widen when volatility increases and narrow when volatility decreases.
     This method allows input of some other form of moving average like EMA or KAMA around which BBAND will be formed.
     Pass desired moving average as <MA> argument. For example BBANDS(MA=TA.KAMA(20)).
    

    "MOBO bands are based on a zone of 0.80 standard deviation with a 10 period look-back"
    If the price breaks out of the MOBO band it can signify a trend move or price spike
    Contains 42% of price movements(noise) within bands.
    
    BBANDS(ohlc, period=10, std_multiplier=0.8, column=column)
    
    BBWIDTH: Bandwidth tells how wide the Bollinger Bands are on a normalized basis.
    %b (pronounced 'percent b') is derived from the formula for Stochastics and shows where price is in relation to the bands.
    %b equals 1 at the upper band and 0 at the lower band.
     
     """
    
    ohlc = ohlc.copy(deep=True)
    ohlc.columns = [c.lower() for c in ohlc.columns]
    
    std = ohlc[column].rolling(window=period).std()

    if not isinstance(MA, pd.Series):
        middle_band = pd.Series( ohlc[column].rolling(window=period).mean(), name="BB_MIDDLE")
    else:
        middle_band = pd.Series(MA, name="BB_MIDDLE")

    upper_bb = pd.Series(middle_band + (std_multiplier * std), name="BB_UPPER")
    lower_bb = pd.Series(middle_band - (std_multiplier * std), name="BB_LOWER")
    
    width_bb = pd.Series((upper_bb - lower_bb)/middle_band, name='BBWITH')
    
    percent_b = pd.Series(
            (ohlc["close"] - lower_bb) / (upper_bb - lower_bb),
            name="pct_b",
        )
    

    return pd.concat([upper_bb, middle_band, lower_bb, width_bb, percent_b], axis=1)

```

##### Calculate MAMA


```python
df = dfs['^GSPC'][['Open', 'High', 'Low', 'Close', 'Volume']]
```


```python
df = df.round(2)
```


```python
cal_bbands
```




    <function __main__.cal_bbands(ohlc: pandas.core.frame.DataFrame, period: int = 20, column: str = 'close', MA: pandas.core.series.Series = None, std_multiplier: float = 2) -> pandas.core.frame.DataFrame>




```python
df_ta = cal_bbands(df, period = 20, column = 'close', std_multiplier=2)
df = df.merge(df_ta, left_index = True, right_index = True, how='inner' )

del df_ta
gc.collect()
```




    22




```python
df_ta = cal_bbands(df, period = 10, column = 'close', std_multiplier=0.8)
df_ta.columns=[f'MOBO_{c}' for c in df_ta.columns]
df = df.merge(df_ta, left_index = True, right_index = True, how='inner' )

del df_ta
gc.collect()
```




    21




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
      <th>BB_UPPER</th>
      <th>BB_MIDDLE</th>
      <th>BB_LOWER</th>
      <th>BBWITH</th>
      <th>pct_b</th>
      <th>MOBO_BB_UPPER</th>
      <th>MOBO_BB_MIDDLE</th>
      <th>MOBO_BB_LOWER</th>
      <th>MOBO_BBWITH</th>
      <th>MOBO_pct_b</th>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>NaN</td>
      <td>NaN</td>
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
      <th>BB_UPPER</th>
      <th>BB_MIDDLE</th>
      <th>BB_LOWER</th>
      <th>BBWITH</th>
      <th>pct_b</th>
      <th>MOBO_BB_UPPER</th>
      <th>MOBO_BB_MIDDLE</th>
      <th>MOBO_BB_LOWER</th>
      <th>MOBO_BBWITH</th>
      <th>MOBO_pct_b</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-08-02</th>
      <td>4104.21</td>
      <td>4140.47</td>
      <td>4079.81</td>
      <td>4091.19</td>
      <td>3880790000</td>
      <td>4151.791900</td>
      <td>3939.3845</td>
      <td>3726.977100</td>
      <td>0.107838</td>
      <td>0.857345</td>
      <td>4083.843144</td>
      <td>4024.452</td>
      <td>3965.060856</td>
      <td>0.029515</td>
      <td>1.061851</td>
    </tr>
    <tr>
      <th>2022-08-03</th>
      <td>4107.96</td>
      <td>4167.66</td>
      <td>4107.96</td>
      <td>4155.17</td>
      <td>3544410000</td>
      <td>4183.001412</td>
      <td>3954.8890</td>
      <td>3726.776588</td>
      <td>0.115357</td>
      <td>0.938996</td>
      <td>4108.592705</td>
      <td>4043.979</td>
      <td>3979.365295</td>
      <td>0.031956</td>
      <td>1.360429</td>
    </tr>
    <tr>
      <th>2022-08-04</th>
      <td>4154.85</td>
      <td>4161.29</td>
      <td>4135.42</td>
      <td>4151.94</td>
      <td>3565810000</td>
      <td>4210.213664</td>
      <td>3967.3550</td>
      <td>3724.496336</td>
      <td>0.122429</td>
      <td>0.880026</td>
      <td>4127.784526</td>
      <td>4059.278</td>
      <td>3990.771474</td>
      <td>0.033753</td>
      <td>1.176301</td>
    </tr>
    <tr>
      <th>2022-08-05</th>
      <td>4115.87</td>
      <td>4151.58</td>
      <td>4107.31</td>
      <td>4145.19</td>
      <td>3540260000</td>
      <td>4232.686012</td>
      <td>3979.6455</td>
      <td>3726.604988</td>
      <td>0.127167</td>
      <td>0.827111</td>
      <td>4143.211043</td>
      <td>4077.634</td>
      <td>4012.056957</td>
      <td>0.032164</td>
      <td>1.015089</td>
    </tr>
    <tr>
      <th>2022-08-08</th>
      <td>4155.93</td>
      <td>4186.62</td>
      <td>4128.97</td>
      <td>4140.06</td>
      <td>3604650000</td>
      <td>4249.440845</td>
      <td>3993.9270</td>
      <td>3738.413155</td>
      <td>0.127951</td>
      <td>0.785959</td>
      <td>4154.042275</td>
      <td>4094.956</td>
      <td>4035.869725</td>
      <td>0.028858</td>
      <td>0.881679</td>
    </tr>
  </tbody>
</table>
</div>



```python
from core.visuals import make_candle
```


```python
df.columns
```




    Index(['Open', 'High', 'Low', 'Close', 'Volume', 'BB_UPPER', 'BB_MIDDLE',
           'BB_LOWER', 'BBWITH', 'pct_b', 'MOBO_BB_UPPER', 'MOBO_BB_MIDDLE',
           'MOBO_BB_LOWER', 'MOBO_BBWITH', 'MOBO_pct_b'],
          dtype='object')




```python

start = -800
end = -200

names = {'main_title': 'Bollinger Bands', 
         'sub_tile': f'{ticker}'}


make_candle(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
            df.iloc[start:end][['BB_UPPER', 'BB_MIDDLE','BB_LOWER']], names = names)
```


    
![png](b_4_1_bbands_intro/output_19_0.png)
    



```python

start = -200
end = -1

names = {'main_title': 'Bollinger Bands', 
         'sub_tile': f'{ticker}'}


make_candle(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
            df.iloc[start:end][['BB_UPPER', 'BB_MIDDLE','BB_LOWER']], names = names)
```


    
![png](b_4_1_bbands_intro/output_20_0.png)
    



```python

start = -200
end = -1

names = {'main_title': 'Bollinger Bands', 
         'sub_tile': f'{ticker}'}


make_candle(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
            df.iloc[start:end][['MOBO_BB_UPPER', 'MOBO_BB_MIDDLE','MOBO_BB_LOWER']], names = names)
```


    
![png](b_4_1_bbands_intro/output_21_0.png)
    



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

def make_3panels(main_data, add_data, mid_panel, chart_type='candle', names=None, figratio=(14,9)):
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
    for name_, data_ in add_data.iteritems():
        added_plots[name_] = mpf.make_addplot(data_)
    
    fb_bbands_ = dict(y1=add_data.iloc[:, 0].values,
                      y2=add_data.iloc[:, 2].values,color="#93c47d",alpha=0.1,interpolate=True)
    fb_bbands_['panel'] = 0

    fb_bbands= [fb_bbands_]
    
    i = 0
    for name_, data_ in mid_panel.iteritems():
        added_plots[name_] = mpf.make_addplot(data_, panel=1, color=all_colors[i])
        i = i + 1
    

    fig, axes = mpf.plot(main_data,  **kwargs,
                         addplot=list(added_plots.values()), 
                         fill_between=fb_bbands)
    # add a new suptitle
    fig.suptitle(names['main_title'], y=1.05, fontsize=12, x=0.285)

    axes[0].legend([None]*(len(added_plots)+len_dict[chart_type]))
    handles = axes[0].get_legend().legendHandles
    axes[0].legend(handles=handles[len_dict[chart_type]:-1],labels=list(added_plots.keys()))
    axes[0].set_title(names['sub_tile'], fontsize=10, style='italic',  loc='left')
    

#     axes[0].set_ylabel(names['y_tiles'][0])
#     axes[2].set_ylabel(names['y_tiles'][1])
    return fig, axes
   
```


```python

start = -150
end = -1

names = {'main_title': 'MOBO Bollinger Bands with Bollinger Bands Width (BBW)', 
         'sub_tile': f'{ticker}'}


aa_, bb_ = make_3panels(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
            df.iloc[start:end][['MOBO_BB_UPPER', 'MOBO_BB_MIDDLE','MOBO_BB_LOWER']], 
             df.iloc[start:end][['MOBO_BBWITH']], 
             chart_type='hollow_and_filled',names = names)
```


    
![png](b_4_1_bbands_intro/output_23_0.png)
    



```python

start = -200
end = -1

names = {'main_title': 'MOBO Bollinger Bands with Bollinger Bands %B', 
         'sub_tile': f'{ticker}'}


aa_, bb_ = make_3panels(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
            df.iloc[start:end][['MOBO_BB_UPPER', 'MOBO_BB_MIDDLE','MOBO_BB_LOWER']], 
             df.iloc[start:end][['MOBO_pct_b']], 
             chart_type='hollow_and_filled',names = names)
```


    
![png](b_4_1_bbands_intro/output_24_0.png)
    



```python

start = -200
end = -1

names = {'main_title': 'Bollinger Bands with Bollinger Bands Width (BBW)', 
         'sub_tile': f'{ticker}'}


aa_, bb_ = make_3panels(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
            df.iloc[start:end][['BB_UPPER', 'BB_MIDDLE','BB_LOWER']], 
             df.iloc[start:end][['BBWITH']], 
             chart_type='hollow_and_filled',names = names)
```


    
![png](b_4_1_bbands_intro/output_25_0.png)
    



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

def make_3panels2(main_data, add_data, mid_panel, chart_type='candle', names=None, figratio=(14,9)):
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
    for name_, data_ in add_data.iteritems():
        added_plots[name_] = mpf.make_addplot(data_)
    
    fb_bbands_ = dict(y1=add_data.iloc[:, 0].values,
                      y2=add_data.iloc[:, 2].values,color="#93c47d",alpha=0.1,interpolate=True)
    fb_bbands_['panel'] = 0
    
    fb_bbands2_ = dict(y1=np.zeros(mid_panel.shape[0]),
                      y2=0.8+np.zeros(mid_panel.shape[0]),color="lightskyblue",alpha=0.1,interpolate=True)
    fb_bbands2_['panel'] = 1

    fb_bbands= [fb_bbands_, fb_bbands2_]
    
    
    i = 0
    for name_, data_ in mid_panel.iteritems():
        added_plots[name_] = mpf.make_addplot(data_, panel=1, color=all_colors[i])
        i = i + 1
    

    fig, axes = mpf.plot(main_data,  **kwargs,
                         addplot=list(added_plots.values()), 
                         fill_between=fb_bbands)
    # add a new suptitle
    fig.suptitle(names['main_title'], y=1.05, fontsize=12, x=0.285)

    axes[0].legend([None]*(len(added_plots)+len_dict[chart_type]))
    handles = axes[0].get_legend().legendHandles
    axes[0].legend(handles=handles[len_dict[chart_type]:-1],labels=list(added_plots.keys()))
    axes[0].set_title(names['sub_tile'], fontsize=10, style='italic',  loc='left')
    

#     axes[0].set_ylabel(names['y_tiles'][0])
#     axes[2].set_ylabel(names['y_tiles'][1])
    return fig, axes
   
```


```python

start = -300
end = -1

names = {'main_title': 'Bollinger Bands w Percent Bandwidth (%B)', 
         'sub_tile': f'{ticker}'}


aa_, bb_ = make_3panels2(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
            df.iloc[start:end][['BB_UPPER', 'BB_MIDDLE','BB_LOWER']], 
             df.iloc[start:end][['pct_b']], 
             chart_type='hollow_and_filled',names = names)
```


    
![png](b_4_1_bbands_intro/output_27_0.png)
    



```python

start = -500
end = -200

names = {'main_title': 'Bollinger Bands w Percent Bandwidth (%B)', 
         'sub_tile': f'{ticker}'}


aa_, bb_ = make_3panels2(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
            df.iloc[start:end][['BB_UPPER', 'BB_MIDDLE','BB_LOWER']], 
             df.iloc[start:end][['pct_b']], 
             chart_type='hollow_and_filled',names = names)
```


    
![png](b_4_1_bbands_intro/output_28_0.png)
    



```python

start = -200
end = -1

names = {'main_title': 'MOBO Bollinger Bands with Bollinger Bands %B', 
         'sub_tile': f'{ticker}'}


aa_, bb_ = make_3panels2(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
            df.iloc[start:end][['MOBO_BB_UPPER', 'MOBO_BB_MIDDLE','MOBO_BB_LOWER']], 
             df.iloc[start:end][['MOBO_pct_b']], 
             chart_type='hollow_and_filled',names = names)
```


    
![png](b_4_1_bbands_intro/output_29_0.png)
    



```python

start = -100
end = -1

names = {'main_title': 'MOBO Bollinger Bands with Bollinger Bands %B', 
         'sub_tile': f'{ticker}'}


aa_, bb_ = make_3panels2(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
            df.iloc[start:end][['MOBO_BB_UPPER', 'MOBO_BB_MIDDLE','MOBO_BB_LOWER']], 
             df.iloc[start:end][['MOBO_pct_b']], 
             chart_type='hollow_and_filled',names = names)
```


    
![png](b_4_1_bbands_intro/output_30_0.png)
    



```python

```
