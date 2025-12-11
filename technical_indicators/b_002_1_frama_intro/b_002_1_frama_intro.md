---
layout: default
title: FRAMA - Fractal Adaptive Moving Average
parent: Technical Indicators
nav_order: 2
---

## FRAMA: Fractal Adaptive Moving Average

**References**

- [prorealcode: Fractal Adaptive Moving Average](https://www.prorealcode.com/prorealtime-indicators/ehlers-fractal-adaptive-moving-average-frama/)
- [FRAMA – Fractal Adaptive Moving Average By John Ehlers](../pdf/FRAMA.pdf)
- [etfhq: Fractal Adaptive Moving Average](http://etfhq.com/blog/2010/09/30/fractal-adaptive-moving-average-frama/)


**Definition**

- The Fractal Adaptive Moving Average (FRAMA) was developed by John Ehlers.
- The indicator is constructed on the EMA exponential moving average algorithm, with a smoothing factor calculated on the basis of the current fractal dimension of the price.
- The advantage of the indicator is the ability to track strong trend movements and market consolidation moments.


**Read the indicator**

The interpretation of the indicator is identical to the interpretation of moving averages
 -  The FRAMA line is relatively “flat” in periods of horizontal range trading. It could therefore be used to avoid many false signals when it is desired to use a technique of the crossing of moving averages.
 -  The FRAMA line has a greater reactivity to changes in trends than moving averages, making it possible to take a much earlier position on a breakout of the horizontal channel.



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
tickers = benchmark_tickers + ['GSK', 'NVO', 'AROC', 'GKOS']
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

    2022-09-10 22:19:15.989643 ^GSPC (5710, 7) 1999-12-31 00:00:00 2022-09-09 00:00:00
    2022-09-10 22:19:16.349383 GSK (5710, 7) 1999-12-31 00:00:00 2022-09-09 00:00:00
    2022-09-10 22:19:16.721121 NVO (5710, 7) 1999-12-31 00:00:00 2022-09-09 00:00:00
    2022-09-10 22:19:16.968170 AROC (3791, 7) 2007-08-21 00:00:00 2022-09-09 00:00:00
    2022-09-10 22:19:17.186939 GKOS (1816, 7) 2015-06-25 00:00:00 2022-09-09 00:00:00
    


```python
ticker = 'GKOS'
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
      <td>49.590000</td>
      <td>50.900002</td>
      <td>48.419998</td>
      <td>48.830002</td>
      <td>650900</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-09-06</th>
      <td>49.200001</td>
      <td>49.200001</td>
      <td>47.630001</td>
      <td>48.099998</td>
      <td>334400</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-09-07</th>
      <td>52.759998</td>
      <td>60.919998</td>
      <td>51.490002</td>
      <td>57.009998</td>
      <td>4560500</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-09-08</th>
      <td>56.439999</td>
      <td>59.599998</td>
      <td>56.439999</td>
      <td>58.380001</td>
      <td>1106900</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-09-09</th>
      <td>58.369999</td>
      <td>58.529999</td>
      <td>55.860001</td>
      <td>56.299999</td>
      <td>1291100</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



##### Define FRAMA calculation function


```python
#https://github.com/peerchemist/finta/blob/master/finta/finta.py
def cal_frama(series: pd.Series, period: int = 16, batch: int=10) -> pd.Series:
        """Fractal Adaptive Moving Average
        :period: Specifies the number of periods used for FRANA calculation
        :batch: Specifies the size of batches used for FRAMA calculation
        """

        assert period % 2 == 0, print("FRAMA period must be even")

        window = batch * 2

        hh = series.rolling(batch).max()
        ll = series.rolling(batch).min()

        n1 = (hh - ll) / batch
        n2 = n1.shift(batch)

        hh2 = series.rolling(window).max()
        ll2 = series.rolling(window).min()
        n3 = (hh2 - ll2) / window

        # calculate fractal dimension
        D = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
        alp = np.exp(-4.6 * (D - 1))
        alp = np.clip(alp, .01, 1).values

        filt = series.values
        for i, x in enumerate(alp):
            cl = series.values[i]
            if i < window:
                continue
            filt[i] = cl * x + (1 - x) * filt[i - 1]

        return pd.Series(filt, index=series.index, name=f"FRAMA{period}")
```

##### Calculate FRAMA


```python
df = dfs[ticker][['Open', 'High', 'Low', 'Close', 'Volume']]
df = df.round(2)
```


```python
df_ta = cal_frama(df['Close'], period = 16, batch = 10)
df = df.merge(df_ta, left_index = True, right_index = True, how='inner' )

del df_ta
gc.collect()
```




    80




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
      <th>FRAMA16</th>
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
      <th>2015-06-25</th>
      <td>29.11</td>
      <td>31.95</td>
      <td>28.00</td>
      <td>31.22</td>
      <td>7554700</td>
      <td>31.22</td>
    </tr>
    <tr>
      <th>2015-06-26</th>
      <td>30.39</td>
      <td>30.39</td>
      <td>27.51</td>
      <td>28.00</td>
      <td>1116500</td>
      <td>28.00</td>
    </tr>
    <tr>
      <th>2015-06-29</th>
      <td>27.70</td>
      <td>28.48</td>
      <td>27.51</td>
      <td>28.00</td>
      <td>386900</td>
      <td>28.00</td>
    </tr>
    <tr>
      <th>2015-06-30</th>
      <td>27.39</td>
      <td>29.89</td>
      <td>27.39</td>
      <td>28.98</td>
      <td>223900</td>
      <td>28.98</td>
    </tr>
    <tr>
      <th>2015-07-01</th>
      <td>28.83</td>
      <td>29.00</td>
      <td>27.87</td>
      <td>28.00</td>
      <td>150000</td>
      <td>28.00</td>
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
      <th>FRAMA16</th>
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
      <th>2022-09-02</th>
      <td>49.59</td>
      <td>50.90</td>
      <td>48.42</td>
      <td>50.322998</td>
      <td>650900</td>
      <td>50.322998</td>
    </tr>
    <tr>
      <th>2022-09-06</th>
      <td>49.20</td>
      <td>49.20</td>
      <td>47.63</td>
      <td>50.267480</td>
      <td>334400</td>
      <td>50.267480</td>
    </tr>
    <tr>
      <th>2022-09-07</th>
      <td>52.76</td>
      <td>60.92</td>
      <td>51.49</td>
      <td>50.951297</td>
      <td>4560500</td>
      <td>50.951297</td>
    </tr>
    <tr>
      <th>2022-09-08</th>
      <td>56.44</td>
      <td>59.60</td>
      <td>56.44</td>
      <td>51.928922</td>
      <td>1106900</td>
      <td>51.928922</td>
    </tr>
    <tr>
      <th>2022-09-09</th>
      <td>58.37</td>
      <td>58.53</td>
      <td>55.86</td>
      <td>52.504161</td>
      <td>1291100</td>
      <td>52.504161</td>
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

def plot_3panels(main_data, add_data=None, mid_panel=None, chart_type='candle', names=None, 
                  figratio=(14,9)):


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
  
        'FRAMA16': mpf.make_addplot(add_data['FRAMA16'], panel=0, color='dodgerblue', secondary_y=False), 
#         'AO-SIGNAL': mpf.make_addplot(mid_panel['AO']-mid_panel['SIGNAL'], type='bar',width=0.7,panel=1, color="pink",alpha=0.65,secondary_y=False),
    }

                         

    fig, axes = mpf.plot(main_data,  **kwargs,
                         addplot=list(added_plots.values()),
                        )
    # add a new suptitle
    fig.suptitle(names['main_title'], y=1.05, fontsize=12, x=0.135)

    axes[0].set_title(names['sub_tile'], fontsize=10, style='italic',  loc='left')
    
    

    
    #set legend

#     axes[0].legend([None]*6)
#     handles = axes[0].get_legend().legendHandles
#     print(handles)
#     axes[0].legend(handles=handles[4:],labels=['MAMA', 'FAMA'])
    #axes[2].set_title('AO', fontsize=10, style='italic',  loc='left')
#     axes[0].set_ylabel('MAMA')
    
    

#     axes[0].set_ylabel(names['y_tiles'][0])
    return fig, axes
   
```


```python

start = -100
end = df.shape[0]

names = {'main_title': f'{ticker}', 
         'sub_tile': 'FRAMA: Fractal Adaptive Moving Average'}


aa_, bb_ = plot_3panels(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
             df.iloc[start:end][[ 'FRAMA16']],
             None, 
             chart_type='hollow_and_filled',
                     names = names, 
                    )
```


    
![png](output_15_0.png)
    

