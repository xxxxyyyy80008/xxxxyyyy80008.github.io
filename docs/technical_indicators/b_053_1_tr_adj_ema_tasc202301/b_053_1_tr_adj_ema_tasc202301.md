---
layout: default
title: True Range Adjusted Exponential Moving Average (TRAdj EMA)
parent: Technical Indicators
nav_order: 53
---

## True Range Adjusted Exponential Moving Average (TRAdj EMA)


**References**


- [traders.com: TradersTips 2023-01](https://traders.com/Documentation/FEEDbk_docs/2023/01/TradersTips.html)



**█ OVERVIEW**


In his article in this issue, “True Range Adjusted Exponential Moving Average (TRAdj EMA),” author Vitali Apirine presents how a security’s true range, which measures volatility, can be incorporated into a traditional exponential moving average. The trend-following indicator, called the true range adjusted exponential moving average (TRAdj EMA), applied with different lengths, can help define turning points and filter price movements. By comparing the indicator with an exponential moving average of identical length, the trader can gain insight into the overall trend.

![png](../img/b_053_1.gif)


![png](../img/b_053_2.gif)



#### Load basic packages 


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
#!pip install yfinance
#!pip install mplfinance
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




    1234



#### Download data


```python
#S&P 500 (^GSPC),  Dow Jones Industrial Average (^DJI), NASDAQ Composite (^IXIC)
#Russell 2000 (^RUT), Crude Oil Nov 21 (CL=F), Gold Dec 21 (GC=F)
#Treasury Yield 10 Years (^TNX)
#CBOE Volatility Index (^VIX) Chicago Options - Chicago Options Delayed Price. Currency in USD

#benchmark_tickers = ['^GSPC', '^DJI', '^IXIC', '^RUT',  'CL=F', 'GC=F', '^TNX']

benchmark_tickers = ['^GSPC', '^VIX']
tickers = benchmark_tickers + ['GSK', 'BST', 'PFE']
```


```python
#https://github.com/ranaroussi/yfinance/blob/main/yfinance/base.py
#     def history(self, period="1mo", interval="1d",
#                 start=None, end=None, prepost=False, actions=True,
#                 auto_adjust=True, back_adjust=False,
#                 proxy=None, rounding=False, tz=None, timeout=None, **kwargs):

dfs = {}

for ticker in tickers:
    cur_data = yf.Ticker(ticker)
    hist = cur_data.history(period="max", start='2000-01-01')
    print(f"{datetime.now()}\t {ticker}\t {hist.shape}\t {hist.index.min()}\t {hist.index.max()}")
    dfs[ticker] = hist
```

    2023-03-05 08:09:41.225347	 ^GSPC	 (5829, 7)	 2000-01-03 00:00:00-05:00	 2023-03-03 00:00:00-05:00
    2023-03-05 08:09:42.065187	 ^VIX	 (5829, 7)	 2000-01-03 00:00:00-05:00	 2023-03-03 00:00:00-05:00
    2023-03-05 08:09:43.042748	 GSK	 (5829, 7)	 2000-01-03 00:00:00-05:00	 2023-03-03 00:00:00-05:00
    2023-03-05 08:09:43.853829	 BST	 (2100, 7)	 2014-10-29 00:00:00-04:00	 2023-03-03 00:00:00-05:00
    2023-03-05 08:09:44.962182	 PFE	 (5829, 7)	 2000-01-03 00:00:00-05:00	 2023-03-03 00:00:00-05:00



```python
ticker = 'GSK'
dfs[ticker].tail(5)
```




<div>
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
      <th>2023-02-27 00:00:00-05:00</th>
      <td>34.900002</td>
      <td>35.080002</td>
      <td>34.759998</td>
      <td>34.820000</td>
      <td>2897500</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-02-28 00:00:00-05:00</th>
      <td>34.549999</td>
      <td>34.669998</td>
      <td>34.270000</td>
      <td>34.270000</td>
      <td>3264100</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-03-01 00:00:00-05:00</th>
      <td>34.220001</td>
      <td>34.349998</td>
      <td>34.060001</td>
      <td>34.259998</td>
      <td>3258500</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-03-02 00:00:00-05:00</th>
      <td>34.139999</td>
      <td>34.639999</td>
      <td>34.099998</td>
      <td>34.580002</td>
      <td>2425200</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-03-03 00:00:00-05:00</th>
      <td>34.450001</td>
      <td>34.669998</td>
      <td>34.389999</td>
      <td>34.660000</td>
      <td>2876300</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Define True Range Adjusted Exponential Moving Average (TRAdj EMA) calculation function



```python
import sys
sys.path.append(r"/kaggle/input/technical-indicators-core")

#from core.finta import TA
from finta import TA
```


```python
df = dfs[ticker][['Open', 'High', 'Low', 'Close', 'Volume']]
df = df.round(2)
```


```python
def cal_tradj_ema(ohlc: pd.DataFrame, 
                  period: int = 40, 
                  adj_period: int = 40, 
                  multiplier: float = 10.0
                 ) -> pd.Series:
    """
    // TASC JAN 2023: True Range Adjusted EMA by Vitali Apirine
    inputs: Periods( 40 ), Pds( 40 ), Mltp( 10 );

    variables:
        Mltp1( 0 ), Mltp2( 0 ),Rate( 0 ),TH( 0 ),TL( 0 ),TR( 0 ),TRAdj( 0 ),TRAdjEMA( 0 );

    Mltp1 = 2 / (Periods + 1);
    TH = Iff(Close[1] > High, Close[1], High);
    TL = Iff(Close[1] < Low, Close[1], Low);
    TR = AbsValue(TH - TL); 
    TRAdj = (TR - Lowest(TR, Pds)) / (Highest(TR, Pds) - Lowest(TR, Pds));
    Mltp2 = TrAdj * Mltp;
    Rate = Mltp1*(1 + Mltp2);

    if CurrentBar = 1 then 
        TRAdjEMA = Close
    else
        TRAdjEMA = TRAdjEMA[1] + Rate * (Close - TRAdjEMA[1]);
    """
    ohlc = ohlc.copy(deep=True)
    ohlc.columns = [c.lower() for c in ohlc.columns]
    
    mltp1 = 2 /(period + 1)
    TH = pd.concat([ohlc['close'].shift(1), ohlc['high']], axis=1).max(axis=1)#true high
    TL = pd.concat([ohlc['close'].shift(1), ohlc['low']], axis=1).min(axis=1) #true low
    TR = (TH-TL).abs()#true range
    
    TR_lowest = TR.rolling(center=False, window=adj_period).min() 
    TR_highest = TR.rolling(center=False, window=adj_period).max()
    TR_Adj = (TR - TR_lowest) / (TR_highest - TR_lowest)
    
    mltp2 = TR_Adj * multiplier
    rate = mltp1 * (1 + mltp2)
    rate.fillna(1, inplace=True)
    
    close = ohlc["close"].copy()        
    _adj_sma =  close.copy()
    
    #for i in range(max(period,adj_period), len(close)):
    for i in range(1, len(close)):
        _adj_sma[i] = _adj_sma[i-1] + rate[i]*(close[i] - _adj_sma[i-1])


    return pd.Series(_adj_sma,index=ohlc.index, name=f"TR_Adj_EMA")   
    
```

#### Calculate True Range Adjusted Exponential Moving Average (TRAdj EMA)


```python
df['TR_Adj_EMA']=cal_tradj_ema(df)
df['EMA']=TA.EMA(df, period = 40, column="close")
```


```python
display(df.head(5))
display(df.tail(5))
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>TR_Adj_EMA</th>
      <th>EMA</th>
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
      <th>2000-01-03 00:00:00-05:00</th>
      <td>19.21</td>
      <td>19.34</td>
      <td>18.89</td>
      <td>19.08</td>
      <td>556100</td>
      <td>19.08</td>
      <td>19.080000</td>
    </tr>
    <tr>
      <th>2000-01-04 00:00:00-05:00</th>
      <td>19.08</td>
      <td>19.08</td>
      <td>18.54</td>
      <td>18.59</td>
      <td>367200</td>
      <td>18.59</td>
      <td>18.828875</td>
    </tr>
    <tr>
      <th>2000-01-05 00:00:00-05:00</th>
      <td>18.84</td>
      <td>19.21</td>
      <td>18.71</td>
      <td>19.21</td>
      <td>481700</td>
      <td>19.21</td>
      <td>18.962320</td>
    </tr>
    <tr>
      <th>2000-01-06 00:00:00-05:00</th>
      <td>19.01</td>
      <td>19.06</td>
      <td>18.54</td>
      <td>18.93</td>
      <td>853800</td>
      <td>18.93</td>
      <td>18.953624</td>
    </tr>
    <tr>
      <th>2000-01-07 00:00:00-05:00</th>
      <td>18.97</td>
      <td>19.92</td>
      <td>18.89</td>
      <td>19.92</td>
      <td>908700</td>
      <td>19.92</td>
      <td>19.166698</td>
    </tr>
  </tbody>
</table>
</div>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>TR_Adj_EMA</th>
      <th>EMA</th>
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
      <th>2023-02-27 00:00:00-05:00</th>
      <td>34.90</td>
      <td>35.08</td>
      <td>34.76</td>
      <td>34.82</td>
      <td>2897500</td>
      <td>35.080205</td>
      <td>35.000215</td>
    </tr>
    <tr>
      <th>2023-02-28 00:00:00-05:00</th>
      <td>34.55</td>
      <td>34.67</td>
      <td>34.27</td>
      <td>34.27</td>
      <td>3264100</td>
      <td>34.918753</td>
      <td>34.964595</td>
    </tr>
    <tr>
      <th>2023-03-01 00:00:00-05:00</th>
      <td>34.22</td>
      <td>34.35</td>
      <td>34.06</td>
      <td>34.26</td>
      <td>3258500</td>
      <td>34.876363</td>
      <td>34.930225</td>
    </tr>
    <tr>
      <th>2023-03-02 00:00:00-05:00</th>
      <td>34.14</td>
      <td>34.64</td>
      <td>34.10</td>
      <td>34.58</td>
      <td>2425200</td>
      <td>34.818843</td>
      <td>34.913140</td>
    </tr>
    <tr>
      <th>2023-03-03 00:00:00-05:00</th>
      <td>34.45</td>
      <td>34.67</td>
      <td>34.39</td>
      <td>34.66</td>
      <td>2876300</td>
      <td>34.809446</td>
      <td>34.900792</td>
    </tr>
  </tbody>
</table>
</div>


#### Visualize True Range Adjusted Exponential Moving Average (TRAdj EMA)


```python
#https://github.com/matplotlib/mplfinance
#this package help visualize financial data
import mplfinance as mpf
import matplotlib.colors as mcolors

# all_colors = list(mcolors.CSS4_COLORS.keys())#"CSS Colors"
# all_colors = list(mcolors.TABLEAU_COLORS.keys()) # "Tableau Palette",
# all_colors = list(mcolors.BASE_COLORS.keys()) #"Base Colors",
all_colors = ['dodgerblue', 'firebrick','limegreen','skyblue','lightgreen',  'navy','yellow','plum',  'yellowgreen']
#https://github.com/matplotlib/mplfinance/issues/181#issuecomment-667252575
#list of colors: https://matplotlib.org/stable/gallery/color/named_colors.html
#https://github.com/matplotlib/mplfinance/blob/master/examples/styles.ipynb

def make_3panels2(main_data, add_data, mid_panel=None, chart_type='candle', names=None, figratio=(14,9)):

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
        
    kwargs = dict(type=chart_type, figratio=figratio, volume=True, volume_panel=1, 
                  panel_ratios=(4,2), tight_layout=True, style=style, returnfig=True)
    
    if names is None:
        names = {'main_title': '', 'sub_tile': ''}
    
    added_plots = { 
#         'S':  mpf.make_addplot(add_data['S'], panel=0, color='blue', type='scatter', marker=r'${S}$' , markersize=100, secondary_y=False),   
#         'B':  mpf.make_addplot(add_data['B'], panel=0, color='blue', type='scatter', marker=r'${B}$' , markersize=100, secondary_y=False), 
        
        'TR_Adj_EMA': mpf.make_addplot(add_data['TR_Adj_EMA'], panel=0, color='dodgerblue', secondary_y=False), 
        'EMA': mpf.make_addplot(add_data['EMA'], panel=0, color='tomato', secondary_y=False), 
    }

        
      
    fb_bbands_ = dict(y1=add_data.iloc[:, 0].values,
                      y2=add_data.iloc[:, 1].values,color="lightskyblue",alpha=0.1,interpolate=True)
    fb_bbands_['panel'] = 0
    

    fb_bbands= [fb_bbands_]
    
    
    if mid_panel is not None:
        i = 0
        for name_, data_ in mid_panel.iteritems():
            added_plots[name_] = mpf.make_addplot(data_, panel=1, color=all_colors[i])
            i = i + 1
        fb_bbands2_ = dict(y1=np.zeros(mid_panel.shape[0]),
                      y2=0.8+np.zeros(mid_panel.shape[0]),color="lightskyblue",alpha=0.1,interpolate=True)
        fb_bbands2_['panel'] = 1
        fb_bbands.append(fb_bbands2_)


    fig, axes = mpf.plot(main_data,  **kwargs,
                         addplot=list(added_plots.values()), 
                         fill_between=fb_bbands)
    # add a new suptitle
    fig.suptitle(names['main_title'], y=1.05, fontsize=12, x=0.1285)

    axes[0].legend([None]*4)
    handles = axes[0].get_legend().legendHandles
    axes[0].legend(handles=handles[2:],labels=['TRAdj EMA', 'EMA'])
    axes[0].set_title(names['sub_tile'], fontsize=10, style='italic',  loc='left')
    

#     axes[0].set_ylabel(names['y_tiles'][0])
#     axes[2].set_ylabel(names['y_tiles'][1])
    return fig, axes
   
```


```python
start = -100
end = df.shape[0]

names = {'main_title': f'{ticker}', 
         'sub_tile': 'True Range Adjusted Exponential Moving Average (TRAdj EMA)'}


aa_, bb_ = make_3panels2(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
            df.iloc[start:end][['TR_Adj_EMA', 'EMA']],
             chart_type='hollow_and_filled',names = names)
```


![png](output_19_0.png)


#### Call the function from finta.py


```python
df_list = []
for ticker, df in dfs.items():
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
    df['TR_Adj_EMA']=TA.TR_Adj_EMA(df)
    df['EMA']=TA.EMA(df, period = 40, column="close")
    df['ticker'] = ticker
    
    df_list.append(df)

```


```python
df_all = pd.concat(df_list)
print(df_all.shape)
del df_list
gc.collect()
```

    (25416, 8)





    6620




```python
dd = df_all.index
df_all.index = dd.date
df_all.index.name='Date'
```


```python
df_all.tail(5)
```




<div>
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>TR_Adj_EMA</th>
      <th>EMA</th>
      <th>ticker</th>
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
      <th>2023-02-27</th>
      <td>41.44</td>
      <td>41.73</td>
      <td>40.73</td>
      <td>40.78</td>
      <td>26370300</td>
      <td>42.998041</td>
      <td>44.684051</td>
      <td>PFE</td>
    </tr>
    <tr>
      <th>2023-02-28</th>
      <td>40.50</td>
      <td>40.71</td>
      <td>40.09</td>
      <td>40.57</td>
      <td>31086900</td>
      <td>42.771926</td>
      <td>44.483366</td>
      <td>PFE</td>
    </tr>
    <tr>
      <th>2023-03-01</th>
      <td>40.56</td>
      <td>40.76</td>
      <td>40.14</td>
      <td>40.18</td>
      <td>21104400</td>
      <td>42.572896</td>
      <td>44.273445</td>
      <td>PFE</td>
    </tr>
    <tr>
      <th>2023-03-02</th>
      <td>40.06</td>
      <td>40.78</td>
      <td>39.81</td>
      <td>40.62</td>
      <td>19110600</td>
      <td>42.263405</td>
      <td>44.095229</td>
      <td>PFE</td>
    </tr>
    <tr>
      <th>2023-03-03</th>
      <td>40.91</td>
      <td>41.18</td>
      <td>40.74</td>
      <td>41.15</td>
      <td>20910700</td>
      <td>42.193500</td>
      <td>43.951559</td>
      <td>PFE</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_all.to_csv('b_053_1_tr_adj_ema_tasc202301.csv', index=True)
```
