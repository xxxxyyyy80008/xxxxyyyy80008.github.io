---
layout: default
title: Moving average bands (MAB) and moving average band width (MABW) 
parent: Technical Indicators
nav_order: 44
---

## Moving average bands (MAB) and moving average band width (MABW) 

**References**


- [traders.com: TradersTips 2021-08](https://traders.com/Documentation/FEEDbk_docs/2021/08/TradersTips.html)
- [Moving Average Bands [CC]](https://www.tradingview.com/script/EjeoPSbI-Moving-Average-Bands-CC/)
- [L1 Vitali Apirine MAB](https://www.tradingview.com/script/DxUYUZxo/)


**█ OVERVIEW**

In “Moving Average Bands” (part 1, July 2021 issue) and “Moving Average Band Width” (part 2, August 2021 issue), author Vitali Apirine explains how moving average bands (MAB) can be used as a trend-following indicator by displaying the movement of a shorter-term moving average in relation to the movement of a longer-term moving average. The distance between the bands will widen as volatility increases and will narrow as volatility decreases. 

In part 2, the moving average band width (MABW) measures the percentage difference between the bands. Changes in this difference may indicate a forthcoming move or change in the trend.



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
    print(datetime.now(), ticker, hist.shape, hist.index.min(), hist.index.max())
    dfs[ticker] = hist
```

    2022-09-04 14:31:18.975062 ^GSPC (5706, 7) 1999-12-31 00:00:00 2022-09-02 00:00:00
    2022-09-04 14:31:19.334500 ^VIX (5706, 7) 1999-12-31 00:00:00 2022-09-02 00:00:00
    2022-09-04 14:31:19.616736 GSK (5706, 7) 1999-12-31 00:00:00 2022-09-02 00:00:00
    2022-09-04 14:31:19.803633 BST (1976, 7) 2014-10-29 00:00:00 2022-09-02 00:00:00
    2022-09-04 14:31:20.100105 PFE (5706, 7) 1999-12-31 00:00:00 2022-09-02 00:00:00
    


```python
ticker = 'PFE'
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
      <th>2022-08-29</th>
      <td>46.380001</td>
      <td>46.689999</td>
      <td>46.119999</td>
      <td>46.230000</td>
      <td>13400500</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-08-30</th>
      <td>46.340000</td>
      <td>46.349998</td>
      <td>45.799999</td>
      <td>45.849998</td>
      <td>16303000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-08-31</th>
      <td>46.009998</td>
      <td>46.290001</td>
      <td>45.130001</td>
      <td>45.230000</td>
      <td>26416800</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-09-01</th>
      <td>45.139999</td>
      <td>46.650002</td>
      <td>45.139999</td>
      <td>46.630001</td>
      <td>19947600</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-09-02</th>
      <td>46.740002</td>
      <td>46.799999</td>
      <td>45.529999</td>
      <td>45.700001</td>
      <td>14662700</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



##### Define Moving average bands (MAB) and moving average band width (MABW)  calculation function





```python
def cal_mabw(ohlc: pd.DataFrame, 
            fast_period: int = 10,
            slow_period: int = 50,
            multiplier: float = 1.0,
            column: str = "close",
        adjust: bool = True 
             
            ) -> pd.DataFrame:
    """
    source: https://traders.com/Documentation/FEEDbk_docs/2021/08/TradersTips.html
    // TASC AUG 2021,  Part 2: A Picture Of Narrowness,  Moving Average Bands (MABW), COPYRIGHT VITALI APIRINE 2021

    inputs:
        int Periods1( 50 ), int Periods2( 10 ),  double Mltp( 1 );

    MA1 = XAverage(Close, Periods1);
    MA2 = XAverage(Close, Periods2);
    Dst = MA1 - MA2;
    DV = Summation(Dst * Dst, Periods2) / Periods2;
    Dev = SquareRoot(DV) * Mltp;
    UpperBand = MA1 + Dev;
    LowerBand = MA1 - Dev;
    BandWidth =(UpperBand - LowerBand) / MA1 * 100;
    LLV = Lowest(BandWidth, Periods1); 
    """
    
    ma1 = ohlc[column].ewm(ignore_na=False, span=slow_period, adjust=adjust).mean()
    ma2 = ohlc[column].ewm(ignore_na=False, span=fast_period, adjust=adjust).mean()
    dst = ma1 - ma2
    #https://stackoverflow.com/questions/60428508/calculating-the-rolling-root-mean-squared
    dv = dst.pow(2).rolling(fast_period).mean().apply(np.sqrt, raw=True)
    dev = dv*multiplier
    
    upper_band = ma1 + dev
    lower_band = ma1 - dev
    middle_band = ma2
    band_width =(upper_band - lower_band) / ma1 * 100
    llv = band_width.rolling(slow_period).min()
   

    return pd.DataFrame(data = {'MAB_UPPER': upper_band.values,
                                'MAB_MIDDLE': middle_band.values, 
                                'MAB_LOWER': lower_band.values, 
                                'MAB_WIDTH': band_width.values,
                                'MAB_LLV': llv.values,
                               },
                        index=ohlc.index, )    
```


```python

```


```python
def cal_mabw(ohlc: pd.DataFrame, 
            fast_period: int = 10,
            slow_period: int = 50,
            multiplier: float = 1.0,
            column: str = "close",
        adjust: bool = True 
             
            ) -> pd.DataFrame:
    """
    source: https://www.tradingview.com/script/EjeoPSbI-Moving-Average-Bands-CC/

    fastLength = input(title="FastLength", type=input.integer, defval=10, minval=1)
    slowLength = input(title="SlowLength", type=input.integer, defval=50, minval=1)
    mult = input(title="Multiple", type=input.integer, defval=1, minval=1)

    fastEma = ema(src, fastLength)
    slowEma = ema(src, slowLength)
    sqAvg = sum(pow(slowEma - fastEma, 2), fastLength) / fastLength
    dev = sqrt(sqAvg) * mult

    upperBand = slowEma + dev
    lowerBand = slowEma - dev
    middleBand = fastEma

    sig = (src > upperBand and nz(src[1]) <= nz(upperBand[1])) or (src > lowerBand and nz(src[1]) <= nz(lowerBand[1]))
          ? 1 : (src < lowerBand and nz(src[1]) >= nz(lowerBand[1])) or (src < upperBand and nz(src[1]) >= nz(upperBand[1])) 
          ? -1 : src > middleBand ? 1 : src < middleBand ? -1 : 0
    alertcondition(crossover(sig, 0), "Buy Signal", "Bullish Change Detected")
    alertcondition(crossunder(sig, 0), "Sell Signal", "Bearish Change Detected")
    mabColor = sig > 0 ? color.green : sig < 0 ? color.red : color.black
    barcolor(bar ? mabColor : na)
    plot(upperBand, title="MabUp", color=mabColor, linewidth=2)
    plot(middleBand, title="MabMid", color=color.black, linewidth=1)
    plot(lowerBand, title="MabLow", color=mabColor, linewidth=2)

    """
    EMA_fast = pd.Series(
        ohlc[column].ewm(ignore_na=False, span=fast_period, adjust=adjust).mean(),
        name="EMA_fast",
    )
    EMA_slow = pd.Series(
        ohlc[column].ewm(ignore_na=False, span=slow_period, adjust=adjust).mean(),
        name="EMA_slow",
    )
    MACD = pd.Series(EMA_fast - EMA_slow, name="MACD")
    #https://stackoverflow.com/questions/60428508/calculating-the-rolling-root-mean-squared
    dev = MACD.pow(2).rolling(fast_period).mean().apply(np.sqrt, raw=True)
    dev = dev*multiplier
    
    upper_band = EMA_slow + dev
    lower_band = EMA_slow - dev
    middle_band = EMA_fast
    band_width =(upper_band - lower_band) / EMA_slow * 100
    llv = band_width.rolling(slow_period).min()
   

    return pd.DataFrame(data = {'MAB_UPPER': upper_band.values, 
                                'MAB_MIDDLE': middle_band.values, 
                                'MAB_LOWER': lower_band.values, 
                                'MAB_WIDTH': band_width.values,
                                'MAB_LLV': llv.values,
                               },
                        index=ohlc.index, )    

```

##### Calculate Moving average bands (MAB) and moving average band width (MABW) 

the 2 functions defined above render exactly same results


```python
df = dfs[ticker][['Open', 'High', 'Low', 'Close', 'Volume']]
```


```python
df = df.round(2)
```


```python
cal_mabw
```




    <function __main__.cal_mabw(ohlc: pandas.core.frame.DataFrame, fast_period: int = 10, slow_period: int = 50, multiplier: float = 1.0, column: str = 'close', adjust: bool = True) -> pandas.core.frame.DataFrame>




```python
df_ta = cal_mabw(df,  fast_period=10, slow_period=50, multiplier=1, column="Close")
df = df.merge(df_ta, left_index = True, right_index = True, how='inner' )

del df_ta
gc.collect()
```




    143




```python
from core.finta import TA
```


```python
df_ta = TA.EMA(df, period = 20, column="close")
df_ta.name='EMA'
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
      <th>MAB_UPPER</th>
      <th>MAB_MIDDLE</th>
      <th>MAB_LOWER</th>
      <th>MAB_WIDTH</th>
      <th>MAB_LLV</th>
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
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1999-12-31</th>
      <td>14.25</td>
      <td>14.31</td>
      <td>14.11</td>
      <td>14.22</td>
      <td>5939817</td>
      <td>NaN</td>
      <td>14.220000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.220000</td>
    </tr>
    <tr>
      <th>2000-01-03</th>
      <td>14.06</td>
      <td>14.20</td>
      <td>13.87</td>
      <td>13.98</td>
      <td>12873345</td>
      <td>NaN</td>
      <td>14.088000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.094000</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>13.70</td>
      <td>13.81</td>
      <td>13.16</td>
      <td>13.46</td>
      <td>14208974</td>
      <td>NaN</td>
      <td>13.835548</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.861199</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>13.54</td>
      <td>13.98</td>
      <td>13.51</td>
      <td>13.68</td>
      <td>12981591</td>
      <td>NaN</td>
      <td>13.784302</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.808890</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>13.70</td>
      <td>14.36</td>
      <td>13.68</td>
      <td>14.17</td>
      <td>11115273</td>
      <td>NaN</td>
      <td>13.895025</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.896239</td>
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
      <th>MAB_UPPER</th>
      <th>MAB_MIDDLE</th>
      <th>MAB_LOWER</th>
      <th>MAB_WIDTH</th>
      <th>MAB_LLV</th>
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
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-08-29</th>
      <td>46.38</td>
      <td>46.69</td>
      <td>46.12</td>
      <td>46.23</td>
      <td>13400500</td>
      <td>50.808139</td>
      <td>47.924590</td>
      <td>48.550520</td>
      <td>4.544383</td>
      <td>0.91289</td>
      <td>48.728796</td>
    </tr>
    <tr>
      <th>2022-08-30</th>
      <td>46.34</td>
      <td>46.35</td>
      <td>45.80</td>
      <td>45.85</td>
      <td>16303000</td>
      <td>50.804365</td>
      <td>47.547391</td>
      <td>48.253953</td>
      <td>5.149314</td>
      <td>0.91289</td>
      <td>48.454625</td>
    </tr>
    <tr>
      <th>2022-08-31</th>
      <td>46.01</td>
      <td>46.29</td>
      <td>45.13</td>
      <td>45.23</td>
      <td>26416800</td>
      <td>50.802100</td>
      <td>47.126048</td>
      <td>47.919030</td>
      <td>5.840838</td>
      <td>0.91289</td>
      <td>48.147518</td>
    </tr>
    <tr>
      <th>2022-09-01</th>
      <td>45.14</td>
      <td>46.65</td>
      <td>45.14</td>
      <td>46.63</td>
      <td>19947600</td>
      <td>50.835359</td>
      <td>47.035857</td>
      <td>47.671609</td>
      <td>6.423404</td>
      <td>0.91289</td>
      <td>48.002992</td>
    </tr>
    <tr>
      <th>2022-09-02</th>
      <td>46.74</td>
      <td>46.80</td>
      <td>45.53</td>
      <td>45.70</td>
      <td>14662700</td>
      <td>50.837723</td>
      <td>46.792974</td>
      <td>47.390541</td>
      <td>7.018717</td>
      <td>0.91289</td>
      <td>47.783659</td>
    </tr>
  </tbody>
</table>
</div>



```python
df[['MAB_LLV', 'MAB_WIDTH']].hist(bins=50)
```




    array([[<AxesSubplot:title={'center':'MAB_LLV'}>,
            <AxesSubplot:title={'center':'MAB_WIDTH'}>]], dtype=object)




    
![png](output_20_1.png)
    



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
        
    kwargs = dict(type=chart_type, figratio=figratio, volume=False, volume_panel=1, 
                  panel_ratios=(4,2), tight_layout=True, style=style, returnfig=True)
    
    if names is None:
        names = {'main_title': '', 'sub_tile': ''}
    
    added_plots = { 
#         'S':  mpf.make_addplot(add_data['S'], panel=0, color='blue', type='scatter', marker=r'${S}$' , markersize=100, secondary_y=False),   
#         'B':  mpf.make_addplot(add_data['B'], panel=0, color='blue', type='scatter', marker=r'${B}$' , markersize=100, secondary_y=False), 
        
        'MAB_UPPER': mpf.make_addplot(add_data['MAB_UPPER'], panel=0, color='dodgerblue', secondary_y=False), 
        'MAB_LOWER': mpf.make_addplot(add_data['MAB_LOWER'], panel=0, color='tomato', secondary_y=False), 
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
    axes[0].legend(handles=handles[2:],labels=['MAB_UPPER', 'MAB_LOWER'])
    axes[0].set_title(names['sub_tile'], fontsize=10, style='italic',  loc='left')
    

#     axes[0].set_ylabel(names['y_tiles'][0])
#     axes[2].set_ylabel(names['y_tiles'][1])
    return fig, axes
   
```


```python

start = -200
end = df.shape[0]

names = {'main_title': f'{ticker}', 
         'sub_tile': 'Moving average bands (MAB) and moving average band width (MABW) '}


aa_, bb_ = make_3panels2(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
            df.iloc[start:end][['MAB_UPPER','MAB_LOWER']],
            df.iloc[start:end][['MAB_WIDTH', 'MAB_LLV']],
             chart_type='hollow_and_filled',names = names)
```


    
![png](output_22_0.png)
    

