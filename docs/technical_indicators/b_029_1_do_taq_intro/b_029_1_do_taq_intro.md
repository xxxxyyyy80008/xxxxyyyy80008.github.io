---
layout: default
title: Donchian Channel
parent: Technical Indicators
nav_order: 29
---

## Donchian Channel

**References**

- [tradingview script: Donchian Channel](https://www.tradingview.com/script/ZZ0T9Wc7-Donchian-Channel-Strategy/)
- [tradingview: Donchian Channel](https://www.tradingview.com/support/solutions/43000502253-donchian-channels-dc/)


Summary
The Donchian Channels indicator (DC) measures volatility in order to gauge whether a market is overbought or oversold. What is important to remember is that Donchian Channels primarily work best within a clearly defined trend. During a Bullish Trend, movement into overbought territory may indicate a strengthening trend (especially if the movements occur frequently during the trend). During a Bearish Trend, frequent movement into oversold territory may also indicate a strengthening trend. Also, during an uptrend (downtrend), movements into oversold (overbought) territory may just be temporary and the overall trend will continue. That being said, being sure of the overall trend plays a major role in getting the most out of Donchian Channels. They could be used with additional technical analysis tools such as trend lines or the Directional Movement (DMI) indicator.
    
**Definition**

Donchian Channels (DC) are used in technical analysis to measure a market's volatility. It is a banded indicator, similar to Bollinger Bands %B (%B). Besides measuring a market's volatility, Donchian Channels are primarily used to identify potential breakouts or overbought/oversold conditions when price reaches either the Upper or Lower Band. These instances would indicate possible trading signals.

The Donchian Channels (DC) indicator was created by the famous commodities trader Richard Donchian. Donchian would become known as The Father of Trend Following.

- Donchian channels are a trend trading channel indicator
- They use an upper and lower band, set either side of a middle line, to identify bearish and bullish rallies, or bearish and bullish reversals
- The upper band is calculated from the highest high in the last n periods
- The lower band is calculated from the lowest low in the last n periods
- The middle line is taken as an average of the current upper band and the current lower band



**Calculation**

---
For this example, a 20 day period is used which is a very commonly used timeframe.

- `Upper Channel = 20 Day High`
- `Lower Channel = 20 Day Low`
- `Middle Channel = (20 Day High + 20 Day Low)/2`

---

**Read the indicator**


Donchian Channels are one of the more straightforward indicators to calculate and understand. The indicator simply takes a user defined number of periods (20 Days for example) and calculates the Upper and Lower Bands. The Upper Band is the high price for the period. The Lower Band is the low price for the period. The Middle Line is simply the average of the two.

The main function of Donchian Channels is to measure volatility. When volatility is high, the bands will widen and when volatility is low, the bands become more narrow. When price reaches or breaks through one of the bands, this indicate an overbought or oversold condition. This can essentially result in one of two things. First, the current trend has been confirmed and the breakthrough will cause a significant move in price in the same direction. The second result is that the trend has already been confirmed and the breakthrough indicates a possible small reversal before continuing in the same direction.


- Trend Confirmation
    - Oftentimes, in a clearly defined trend, overbought or oversold conditions can be a significant sign of strength.
    - During a Bullish Trend price moving into overbought territory can indicate a strengthening trend.
    - During a Bearish Trend price moving into oversold territory can indicate a strengthening trend.

- Overbought/Oversold
    - During a Bullish Trend, price may move into short-term oversold conditions. These oversold conditions can also indicate a strengthening trend.
    - During a Bearish Trend, price may move into short-term overbought conditions. These overbought conditions can also indicate a strengthening trend.


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
tickers = benchmark_tickers + ['GSK', 'NVO', 'AROC', 'RETA']
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

    2022-09-10 21:18:25.967422 ^GSPC (5710, 7) 1999-12-31 00:00:00 2022-09-09 00:00:00
    2022-09-10 21:18:26.243235 GSK (5710, 7) 1999-12-31 00:00:00 2022-09-09 00:00:00
    2022-09-10 21:18:26.524217 NVO (5710, 7) 1999-12-31 00:00:00 2022-09-09 00:00:00
    2022-09-10 21:18:26.757792 AROC (3791, 7) 2007-08-21 00:00:00 2022-09-09 00:00:00
    2022-09-10 21:18:26.916846 RETA (1584, 7) 2016-05-26 00:00:00 2022-09-09 00:00:00
    


```python
ticker = 'GSK'
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
      <th>2022-09-02</th>
      <td>31.600000</td>
      <td>31.969999</td>
      <td>31.469999</td>
      <td>31.850000</td>
      <td>8152600</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-09-06</th>
      <td>31.650000</td>
      <td>31.760000</td>
      <td>31.370001</td>
      <td>31.469999</td>
      <td>5613900</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-09-07</th>
      <td>31.209999</td>
      <td>31.590000</td>
      <td>31.160000</td>
      <td>31.490000</td>
      <td>4822000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-09-08</th>
      <td>30.910000</td>
      <td>31.540001</td>
      <td>30.830000</td>
      <td>31.510000</td>
      <td>6620900</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-09-09</th>
      <td>31.950001</td>
      <td>31.969999</td>
      <td>31.730000</td>
      <td>31.889999</td>
      <td>3556800</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



##### Define Donchian Channel calculation function



```python

#https://github.com/peerchemist/finta/blob/af01fa594995de78f5ada5c336e61cd87c46b151/finta/finta.py
#https://github.com/mpquant/MyTT/blob/ea4f14857ecc46a3739a75ce2e6974b9057a6102/MyTT.py

#   TAQ
#https://github.com/mpquant/MyTT/blob/ea4f14857ecc46a3739a75ce2e6974b9057a6102/MyTT.py


def cal_taq(ohlc: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    
    """
   TAQ
    
    """
    
    ohlc = ohlc.copy(deep=True)
    ohlc.columns =  [c.lower() for c in ohlc.columns]
    
    highest_high = ohlc["high"].rolling(center=False, window=period).max()
    lowest_low = ohlc["low"].rolling(center=False, window=period).min() 
    mid = (highest_high + lowest_low)/2    
    
    
    return pd.DataFrame(data={'TAQ_HH': highest_high.values, 
                           'TAQ_LL': lowest_low.values, 
                           'TAQ_MID': mid.values,
                             }, 
                     index=ohlc.index)


def cal_do(
    ohlc: pd.DataFrame, slow_period: int = 20, fast_period: int = 5
) -> pd.DataFrame:
    """Donchian Channel, a moving average indicator developed by Richard Donchian.
    It plots the highest high and lowest low over the last period time intervals.
    
    source: https://github.com/mpquant/MyTT/blob/ea4f14857ecc46a3739a75ce2e6974b9057a6102/MyTT.py

        def TAQ(HIGH,LOW,N):                               #唐安奇通道(海龟)交易指标，大道至简，能穿越牛熊
            UP=HHV(HIGH,N);    DOWN=LLV(LOW,N);    MID=(UP+DOWN)/2
            return UP,MID,DOWN    
    
    """
    
        
    ohlc = ohlc.copy(deep=True)
    ohlc.columns =  [c.lower() for c in ohlc.columns]

    upper = pd.Series(
        ohlc["high"].rolling(center=False, window=slow_period).max(), name="DO_UPPER"
    )
    lower = pd.Series(
        ohlc["low"].rolling(center=False, window=fast_period).min(), name="DO_LOWER"
    )
    middle = pd.Series((upper + lower) / 2, name="DO_MIDDLE")

    return pd.concat([lower, middle, upper], axis=1)
```

##### Calculate Donchian Channel


```python
df = dfs[ticker][['Open', 'High', 'Low', 'Close', 'Volume']]
```


```python
df = df.round(2)
```


```python
cal_do
```




    <function __main__.cal_do(ohlc: pandas.core.frame.DataFrame, slow_period: int = 20, fast_period: int = 5) -> pandas.core.frame.DataFrame>




```python
# df_ta = cal_do(df, slow_period = 20, fast_period = 7)
df_ta = cal_do(df, slow_period = 20, fast_period = 20)
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
      <th>DO_LOWER</th>
      <th>DO_MIDDLE</th>
      <th>DO_UPPER</th>
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
      <td>19.60</td>
      <td>19.67</td>
      <td>19.52</td>
      <td>19.56</td>
      <td>139400</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-03</th>
      <td>19.58</td>
      <td>19.71</td>
      <td>19.25</td>
      <td>19.45</td>
      <td>556100</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>19.45</td>
      <td>19.45</td>
      <td>18.90</td>
      <td>18.95</td>
      <td>367200</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>19.21</td>
      <td>19.58</td>
      <td>19.08</td>
      <td>19.58</td>
      <td>481700</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>19.38</td>
      <td>19.43</td>
      <td>18.90</td>
      <td>19.30</td>
      <td>853800</td>
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
      <th>DO_LOWER</th>
      <th>DO_MIDDLE</th>
      <th>DO_UPPER</th>
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
      <th>2022-09-02</th>
      <td>31.60</td>
      <td>31.97</td>
      <td>31.47</td>
      <td>31.85</td>
      <td>8152600</td>
      <td>31.47</td>
      <td>35.835</td>
      <td>40.20</td>
    </tr>
    <tr>
      <th>2022-09-06</th>
      <td>31.65</td>
      <td>31.76</td>
      <td>31.37</td>
      <td>31.47</td>
      <td>5613900</td>
      <td>31.37</td>
      <td>35.625</td>
      <td>39.88</td>
    </tr>
    <tr>
      <th>2022-09-07</th>
      <td>31.21</td>
      <td>31.59</td>
      <td>31.16</td>
      <td>31.49</td>
      <td>4822000</td>
      <td>31.16</td>
      <td>35.025</td>
      <td>38.89</td>
    </tr>
    <tr>
      <th>2022-09-08</th>
      <td>30.91</td>
      <td>31.54</td>
      <td>30.83</td>
      <td>31.51</td>
      <td>6620900</td>
      <td>30.83</td>
      <td>33.285</td>
      <td>35.74</td>
    </tr>
    <tr>
      <th>2022-09-09</th>
      <td>31.95</td>
      <td>31.97</td>
      <td>31.73</td>
      <td>31.89</td>
      <td>3556800</td>
      <td>30.83</td>
      <td>33.285</td>
      <td>35.74</td>
    </tr>
  </tbody>
</table>
</div>



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

def make_3panels2(main_data, add_data, mid_panel=None, chart_type='candle', names=None, figratio=(14,9)):
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
        
    kwargs = dict(type=chart_type, figratio=figratio, volume=True, volume_panel=1, 
                  panel_ratios=(4,2), tight_layout=True, style=style, returnfig=True)
    
    if names is None:
        names = {'main_title': '', 'sub_tile': ''}


    added_plots = { 
        'DO_UPPER':  mpf.make_addplot(add_data['DO_UPPER'], panel=0, color='dodgerblue', width=1, secondary_y=False), 
        'DO_LOWER':  mpf.make_addplot(add_data['DO_LOWER'], panel=0, color='orange', width=1, secondary_y=False), 
        'DO_MIDDLE':  mpf.make_addplot(add_data['DO_MIDDLE'], panel=0, color='green', width=1, secondary_y=False), 
    }

    
    fb_bbands_ = dict(y1=add_data.iloc[:, 0].values,
                      y2=add_data.iloc[:, 2].values,color="lightskyblue",alpha=0.1,interpolate=True)
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

    axes[0].legend([None]*6)
    handles = axes[0].get_legend().legendHandles
    axes[0].legend(handles=handles[2:],labels=list(added_plots.keys()))
    axes[0].set_title(names['sub_tile'], fontsize=10, style='italic',  loc='left')
    

#     axes[0].set_ylabel(names['y_tiles'][0])
#     axes[2].set_ylabel(names['y_tiles'][1])
    return fig, axes
   
```


```python
start = -100
end = df.shape[0]


names = {'main_title': f'{ticker}', 
         'sub_tile': 'Donchian Channel'}


aa_, bb_ = make_3panels2(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
            df.iloc[start:end][['DO_UPPER', 'DO_MIDDLE','DO_LOWER']],
             chart_type='hollow_and_filled',names = names)
```


    
![png](output_17_0.png)
    

