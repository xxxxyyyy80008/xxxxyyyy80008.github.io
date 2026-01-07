---
layout: default
title: BRAR
parent: Technical Indicators
nav_order: 35
---

## BRAR

**References**

- [sentiment-indicators](https://www.netinbag.com/en/finance/in-finance-what-are-sentiment-indicators.html)
- [pandas-ta: brar](https://github.com/twopirllc/pandas-ta/blob/2a24fdc1b69110332db39eda9723a628f75eaf7a/pandas_ta/momentum/brar.py)
- [github:MyTT](https://github.com/mpquant/MyTT/blob/ea4f14857ecc46a3739a75ce2e6974b9057a6102/MyTT.py)

**Definition**

Emotional index (BRAR) is also called popularity intention index. It consists of two indicators: popularity index (AR) and willingness index (BR). Both the AR indicator and the BR indicator are technical indicators based on the analysis of historical stock prices.

- the BRAR indicator is 100-centric. When BR around 100 indicates the sentiment of the market is in a very balanced state. 
- when the BRAR starts to fluctuate, it can rise above 200 or drop below 80. 



**Read the indicator**

- AR indicator can be used alone, and BR indicator needs to be used in conjunction with AR indicators in order to be effective. 
- BRAR is not suitable for capturing a large bottom, but it can be used to capture a local bottom.


Buy signals:

- BR line normally runs above AR line, when BR crosses AR and runs below AR line
- BR<40 and AR<60
- BR <AR and AR <50
- BR <AR and BR <100


Sell signals

- BR>400 and AR>180
- BR rapadily increases but AR stays flat or slightly drops
       




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
tickers = benchmark_tickers + ['GSK', 'NVO', 'GKOS']
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

    2022-09-10 21:31:52.358741 ^GSPC (5710, 7) 1999-12-31 00:00:00 2022-09-09 00:00:00
    2022-09-10 21:31:52.676873 GSK (5710, 7) 1999-12-31 00:00:00 2022-09-09 00:00:00
    2022-09-10 21:31:53.016940 NVO (5710, 7) 1999-12-31 00:00:00 2022-09-09 00:00:00
    2022-09-10 21:31:53.171193 GKOS (1816, 7) 2015-06-25 00:00:00 2022-09-09 00:00:00
    


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



##### Define BRAR  calculation function
       
        
    def BRAR(OPEN,CLOSE,HIGH,LOW,M1=26):                 #BRAR-ARBR 情绪指标  
        AR = SUM(HIGH - OPEN, M1) / SUM(OPEN - LOW, M1) * 100
        BR = SUM(MAX(0, HIGH - REF(CLOSE, 1)), M1) / SUM(MAX(0, REF(CLOSE, 1) - LOW), M1) * 100
        return AR, BR        
        


```python
def cal_brar(ohlc: pd.DataFrame, period: int = 26) -> pd.DataFrame:
    """
    BUY: AR<60 BR<40
    SELL: BR>400, AR>180
    
    reference:
    
    https://github.com/mpquant/MyTT/blob/ea4f14857ecc46a3739a75ce2e6974b9057a6102/MyTT.py
    https://github.com/twopirllc/pandas-ta/blob/2a24fdc1b69110332db39eda9723a628f75eaf7a/pandas_ta/momentum/brar.py
    
    """
    
    ohlc = ohlc.copy()
    ohlc.columns = [c.lower() for c in ohlc.columns]    
    
    h, l, o, c = ohlc["high"], ohlc["low"], ohlc["open"], ohlc["close"]
    c1 = c.shift(1)
    
    a0 = (h-o).rolling(period).sum()
    a1 = (o-l).rolling(period).sum()
    ar = a0/a1*100
    
    b0 = (h - c1).apply(lambda x: max(0, x)).rolling(period).sum()
    b1 = (c1 - l).apply(lambda x: max(0, x)).rolling(period).sum()
    br = b0/b1*100
    
    return pd.DataFrame(data={'AR': ar.values, 'BR': br.values}, index=ohlc.index)
```

##### Calculate BRAR


```python
df = dfs[ticker][['Open', 'High', 'Low', 'Close', 'Volume']]
```


```python
df = df.round(2)
```


```python
help(cal_brar)
```

    Help on function cal_brar in module __main__:
    
    cal_brar(ohlc: pandas.core.frame.DataFrame, period: int = 26) -> pandas.core.frame.DataFrame
        BUY: AR<60 BR<40
        SELL: BR>400, AR>180
        
        reference:
        
        https://github.com/mpquant/MyTT/blob/ea4f14857ecc46a3739a75ce2e6974b9057a6102/MyTT.py
        https://github.com/twopirllc/pandas-ta/blob/2a24fdc1b69110332db39eda9723a628f75eaf7a/pandas_ta/momentum/brar.py
    
    


```python
df_ta = cal_brar(df, period = 14)
df = df.merge(df_ta, left_index = True, right_index = True, how='inner' )

del df_ta
gc.collect()
```




    122




```python
#     BUY: AR<60 BR<40
#     SELL: BR>400, AR>180

df['B'] = ((df["AR"]<60) & (df["BR"]<40)).astype(int)*(df['High']+df['Low'])/2
df['S'] = ((df["AR"]>180) & (df["BR"]>400)).astype(int)*(df['High']+df['Low'])/2
```


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
      <th>AR</th>
      <th>BR</th>
      <th>B</th>
      <th>S</th>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-26</th>
      <td>30.39</td>
      <td>30.39</td>
      <td>27.51</td>
      <td>28.00</td>
      <td>1116500</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-29</th>
      <td>27.70</td>
      <td>28.48</td>
      <td>27.51</td>
      <td>28.00</td>
      <td>386900</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-30</th>
      <td>27.39</td>
      <td>29.89</td>
      <td>27.39</td>
      <td>28.98</td>
      <td>223900</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-07-01</th>
      <td>28.83</td>
      <td>29.00</td>
      <td>27.87</td>
      <td>28.00</td>
      <td>150000</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <th>AR</th>
      <th>BR</th>
      <th>B</th>
      <th>S</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-09-02</th>
      <td>49.59</td>
      <td>50.90</td>
      <td>48.42</td>
      <td>48.83</td>
      <td>650900</td>
      <td>48.783455</td>
      <td>57.727550</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-09-06</th>
      <td>49.20</td>
      <td>49.20</td>
      <td>47.63</td>
      <td>48.10</td>
      <td>334400</td>
      <td>45.229469</td>
      <td>59.610553</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-09-07</th>
      <td>52.76</td>
      <td>60.92</td>
      <td>51.49</td>
      <td>57.01</td>
      <td>4560500</td>
      <td>96.770186</td>
      <td>161.901306</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-09-08</th>
      <td>56.44</td>
      <td>59.60</td>
      <td>56.44</td>
      <td>58.38</td>
      <td>1106900</td>
      <td>116.763754</td>
      <td>187.993921</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-09-09</th>
      <td>58.37</td>
      <td>58.53</td>
      <td>55.86</td>
      <td>56.30</td>
      <td>1291100</td>
      <td>110.436893</td>
      <td>180.231716</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
df[['AR','BR']].hist(bins=50)
```




    array([[<AxesSubplot:title={'center':'AR'}>,
            <AxesSubplot:title={'center':'BR'}>]], dtype=object)




    
![png](output_17_1.png)
    



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
        
    kwargs = dict(type=chart_type, figratio=figratio, volume=False,
                  panel_ratios=(4,2), tight_layout=True, style=style, returnfig=True)
    
    if names is None:
        names = {'main_title': '', 'sub_tile': ''}
    


    added_plots = { 
        'S':  mpf.make_addplot(add_data['S'], panel=0, color='blue', type='scatter', marker=r'${S}$' , markersize=100, secondary_y=False),   
        'B':  mpf.make_addplot(add_data['B'], panel=0, color='blue', type='scatter', marker=r'${B}$' , markersize=100, secondary_y=False), 
        
        'AR': mpf.make_addplot(mid_panel['AR'], panel=1, color='dodgerblue', secondary_y=False), 
        'BR': mpf.make_addplot(mid_panel['BR'], panel=1, color='tomato', secondary_y=False), 
#         'AO-SIGNAL': mpf.make_addplot(mid_panel['AO']-mid_panel['SIGNAL'], type='bar',width=0.7,panel=1, color="pink",alpha=0.65,secondary_y=False),
    }

                         

    fig, axes = mpf.plot(main_data,  **kwargs,
                         addplot=list(added_plots.values()),
                        )
    # add a new suptitle
    fig.suptitle(names['main_title'], y=1.05, fontsize=12, x=0.128)

    axes[0].set_title(names['sub_tile'], fontsize=10, style='italic',  loc='left')
    
    

    
    #set legend

    axes[2].legend([None]*2)
    handles = axes[2].get_legend().legendHandles
#     print(handles)
    axes[2].legend(handles=handles,labels=['AR', 'BR'])
    #axes[2].set_title('AO', fontsize=10, style='italic',  loc='left')
    axes[2].set_ylabel('BRAR')
    
    

#     axes[0].set_ylabel(names['y_tiles'][0])
    return fig, axes
   
```


```python

start = -100
end = df.shape[0]

names = {'main_title': f'{ticker}', 
         'sub_tile': 'BRAR: BUY: AR<60 BR<40; SELL: BR>400, AR>180'}


aa_, bb_ = plot_3panels(df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume']], 
             df.iloc[start:end][['B', 'S']],
             df.iloc[start:end][['AR', 'BR']], 
             chart_type='hollow_and_filled',
                     names = names, 
                    )
```


    
![png](output_19_0.png)
    

