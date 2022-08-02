## DFL Exploratory Data Analysis and Video Processing 

###  Part I: install packages for video data processing and analysis

- packages for video data analysis: `imageio` package and `moviepy` package
- `imageio` package
    - `imageio` package does not require additional installation. Kaggle notebook currently have `imageio==2.19.3` 
    -  however, a `imageio-ffmpeg`, which is a required dependency for video data processing, needs to be installed
    -  follow the below steps to install `imageio-ffmpeg` offline
    >    - click `+ Add data`  button on the upper right corner of your notebook
    >    - click on `Notebook Output Files`
    >    - enter `imageio-ffmpeg` in the search box and click search
    >    - you will see that **imageio-ffmpeg** notebook on the top of the list, click `Add`
    >    - copy `!conda install /kaggle/input/imageio-ffmpeg/*.tar.bz2` to a cell and run it
- `moviepy` pakcage
    - similar to `imageio-ffmpeg` package, additional effort is required to install the package offline
    - follow the below steps to install it
    >    - click `+ Add data`  button on the upper right corner of your notebook
    >    - click on `Notebook Output Files`
    >    - enter `moviepy` in the search box and click search
    >    - you will see that **moviepy** notebook on the top of the list, click `Add`
    >    - copy `!conda install /kaggle/input/moviepy/*.tar.bz2` to a cell and run it

### Part II: exploratory data analysis for video data

#### Understand the train data
- task description: detect three kinds of player events, both the time of occurrence and the type, within these videos. 
- three kinds of player events: 

>Plays: A Play describes a player’s attempt to switch ball control to another member of his team. A play event may be executed as a Pass or as a Cross.

>Throw-Ins: A Throw-In refers to a situation where the game is restarted after the ball went out of play over the sideline following the touch of the opposite team. The ball must be thrown with hands, from behind and over the head of executing player.

> Challenge: A Challenge is a player action during which two players of opposing teams are physically capable of either gaining or receiving ball control and attempt to do so. A Challenge requires one of the two players to touch the ball or to foul the opposing player.

- Training Data
    - train/ - Folder containing videos to be used as training data, comprising video recordings from eight games. 
    - train.csv - Event annotations for videos in the train/ folder.
        - video_id - Identifies which video the event occurred in.
        - event - The type of event occurrence, one of challenge, play, or throwin. Also present are labels start and end indicating the scoring intervals of the video. 
        - event_attributes - Additional descriptive attributes for the event.
        - time - The time, in seconds, the event occurred within the video.
- Understand the train data
    - in train data, there are 4382 samples of event data (play, throwin, challenge), 3418 start timestamps and 3418 end timestamps for events. 
    - among the 4382 event samples: 81% are `play` events, only ~4% of samples are `throwin` events
    - the gap between *start timestamp* of an event and the *timestamp of the event* can be as long as 2 seconds and as short as half a second
    - the gap between *end timestamp* of an event and the *timestamp of the event* can be as long as 2 seconds and as short as half a second
    - the gap between *start timestamp* of an event and *end timestamp* of the event, when both timestamps present, are around 2.5 seconds
    
#### Explore the video data







### references:
- https://imageio.readthedocs.io/en/stable/examples.html#iterate-over-frames-in-a-movie
- https://imageio.readthedocs.io/en/latest/reference/userapi.html?highlight=get_data#imageio.core.format.Reader.get_meta_data
- https://stackoverflow.com/questions/72773615/how-to-seek-a-frame-in-video-using-imageio
- https://stackoverflow.com/questions/52257731/extract-part-of-a-video-using-ffmpeg-extract-subclip-black-frames
- https://stackoverflow.com/questions/29718238/how-to-read-mp4-video-to-be-processed-by-scikit-image
- https://zulko.github.io/moviepy/ref/ffmpeg.html?highlight=ffmpeg_extract_subclip#moviepy.video.io.ffmpeg_tools.ffmpeg_extract_subclip


####  install packages for video data processing and analysis


```python
!conda install /kaggle/input/imageio-ffmpeg/*.tar.bz2
```

    
    Downloading and Extracting Packages
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    


```python
!conda install /kaggle/input/moviepy/*.tar.bz2
```

    
    Downloading and Extracting Packages
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    ######################################################################## | 100% 
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    

### load packages


```python
#basic libs

import pandas as pd
import numpy as np
import os
from pathlib import Path

from datetime import datetime, timedelta
import time
from dateutil.relativedelta import relativedelta

import gc
import copy


#additional data processing


from sklearn.preprocessing import StandardScaler, MinMaxScaler


#visualization
import seaborn as sns
import matplotlib.pyplot as plt

#load images
import matplotlib.image as mpimg
import PIL
from PIL import Image

#for loading videos
import imageio
import imageio.v2 as iio
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip  #to extract a sub clip from a video
from IPython.display import Video #to play video in notebook


#settings

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

Image.MAX_IMAGE_PIXELS = None

import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
random_seed=1234
pl.seed_everything(random_seed)


```




    1234



#### Load and understand train data



```python
df_train = pd.read_csv("/kaggle/input/dfl-bundesliga-data-shootout/train.csv")
df_train.shape
```




    (11218, 4)




```python
df_train.head(2)
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
      <th>video_id</th>
      <th>time</th>
      <th>event</th>
      <th>event_attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1606b0e6_0</td>
      <td>200.265822</td>
      <td>start</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1606b0e6_0</td>
      <td>201.150000</td>
      <td>challenge</td>
      <td>['ball_action_forced']</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_train['event'].value_counts()
```




    play         3586
    start        3418
    end          3418
    challenge     624
    throwin       172
    Name: event, dtype: int64




```python
df_train['event_attributes'].value_counts()
```




    ['pass', 'openplay']                  3337
    ['ball_action_forced']                 239
    ['pass']                               154
    ['opponent_dispossessed']              138
    ['pass', 'freekick']                   127
    ['fouled']                             111
    ['cross', 'openplay']                   80
    ['challenge_during_ball_transfer']      53
    ['possession_retained']                 44
    ['opponent_rounded']                    39
    ['cross', 'corner']                     33
    ['cross']                               18
    ['cross', 'freekick']                    5
    ['pass', 'corner']                       4
    Name: event_attributes, dtype: int64




```python
df_train['video_id'].value_counts()
```




    1606b0e6_1    1249
    35bd9041_0    1075
    3c993bd2_0    1042
    1606b0e6_0    1000
    ecf251d4_0     980
    3c993bd2_1     966
    35bd9041_1     933
    407c5a9e_1     858
    cfbe2e94_0     823
    4ffd5986_0     792
    cfbe2e94_1     763
    9a97dae4_1     737
    Name: video_id, dtype: int64




```python
df_train[df_train['video_id']=='1606b0e6_0'].head(10)
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
      <th>video_id</th>
      <th>time</th>
      <th>event</th>
      <th>event_attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1606b0e6_0</td>
      <td>200.265822</td>
      <td>start</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1606b0e6_0</td>
      <td>201.150000</td>
      <td>challenge</td>
      <td>['ball_action_forced']</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1606b0e6_0</td>
      <td>202.765822</td>
      <td>end</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1606b0e6_0</td>
      <td>210.124111</td>
      <td>start</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1606b0e6_0</td>
      <td>210.870000</td>
      <td>challenge</td>
      <td>['opponent_dispossessed']</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1606b0e6_0</td>
      <td>212.624111</td>
      <td>end</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1606b0e6_0</td>
      <td>217.850213</td>
      <td>start</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1606b0e6_0</td>
      <td>219.230000</td>
      <td>throwin</td>
      <td>['pass']</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1606b0e6_0</td>
      <td>220.350213</td>
      <td>end</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1606b0e6_0</td>
      <td>223.930850</td>
      <td>start</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_train[df_train['video_id']=='1606b0e6_0']['event'].value_counts()
```




    play         319
    start        302
    end          302
    challenge     56
    throwin       21
    Name: event, dtype: int64




```python
df_train.sort_values(by=['video_id', 'time'], ascending=[True, True], inplace=True)
df_train['seq'] = 1
df_train['seq'] = df_train[['video_id', 'seq']].groupby('video_id').cumsum()
```


```python
start_df = df_train[df_train['event']=='start']
end_df = df_train[df_train['event']=='end']
event_df = df_train[~df_train['event'].isin(['start', 'end'])]
print(start_df.shape, end_df.shape, event_df.shape)

start_df.columns = [f's_{c}' for c in start_df.columns]
end_df.columns = [f'e_{c}' for c in end_df.columns]
display(start_df.head(2))
display(end_df.head(2))

event_df['s_seq'] = event_df['seq'] - 1
event_df['e_seq'] = event_df['seq'] + 1

display(event_df.head(2))
```

    (3418, 5) (3418, 5) (4382, 5)
    


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
      <th>s_video_id</th>
      <th>s_time</th>
      <th>s_event</th>
      <th>s_event_attributes</th>
      <th>s_seq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1606b0e6_0</td>
      <td>200.265822</td>
      <td>start</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1606b0e6_0</td>
      <td>210.124111</td>
      <td>start</td>
      <td>NaN</td>
      <td>4</td>
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
      <th>e_video_id</th>
      <th>e_time</th>
      <th>e_event</th>
      <th>e_event_attributes</th>
      <th>e_seq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1606b0e6_0</td>
      <td>202.765822</td>
      <td>end</td>
      <td>NaN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1606b0e6_0</td>
      <td>212.624111</td>
      <td>end</td>
      <td>NaN</td>
      <td>6</td>
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
      <th>video_id</th>
      <th>time</th>
      <th>event</th>
      <th>event_attributes</th>
      <th>seq</th>
      <th>s_seq</th>
      <th>e_seq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1606b0e6_0</td>
      <td>201.15</td>
      <td>challenge</td>
      <td>['ball_action_forced']</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1606b0e6_0</td>
      <td>210.87</td>
      <td>challenge</td>
      <td>['opponent_dispossessed']</td>
      <td>5</td>
      <td>4</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



```python
print(event_df.shape)
display(event_df.head(2))
event_df = event_df.merge(start_df, left_on=['video_id', 's_seq'], right_on=['s_video_id', 's_seq'], how='left')
print(event_df.shape)
display(event_df.head(2))
event_df = event_df.merge(end_df, left_on=['video_id', 'e_seq'], right_on=['e_video_id', 'e_seq'], how='left')
print(event_df.shape)
display(event_df.head(2))
```

    (4382, 7)
    


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
      <th>video_id</th>
      <th>time</th>
      <th>event</th>
      <th>event_attributes</th>
      <th>seq</th>
      <th>s_seq</th>
      <th>e_seq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1606b0e6_0</td>
      <td>201.15</td>
      <td>challenge</td>
      <td>['ball_action_forced']</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1606b0e6_0</td>
      <td>210.87</td>
      <td>challenge</td>
      <td>['opponent_dispossessed']</td>
      <td>5</td>
      <td>4</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>


    (4382, 11)
    


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
      <th>video_id</th>
      <th>time</th>
      <th>event</th>
      <th>event_attributes</th>
      <th>seq</th>
      <th>s_seq</th>
      <th>e_seq</th>
      <th>s_video_id</th>
      <th>s_time</th>
      <th>s_event</th>
      <th>s_event_attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1606b0e6_0</td>
      <td>201.15</td>
      <td>challenge</td>
      <td>['ball_action_forced']</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1606b0e6_0</td>
      <td>200.265822</td>
      <td>start</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1606b0e6_0</td>
      <td>210.87</td>
      <td>challenge</td>
      <td>['opponent_dispossessed']</td>
      <td>5</td>
      <td>4</td>
      <td>6</td>
      <td>1606b0e6_0</td>
      <td>210.124111</td>
      <td>start</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    (4382, 15)
    


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
      <th>video_id</th>
      <th>time</th>
      <th>event</th>
      <th>event_attributes</th>
      <th>seq</th>
      <th>s_seq</th>
      <th>e_seq</th>
      <th>s_video_id</th>
      <th>s_time</th>
      <th>s_event</th>
      <th>s_event_attributes</th>
      <th>e_video_id</th>
      <th>e_time</th>
      <th>e_event</th>
      <th>e_event_attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1606b0e6_0</td>
      <td>201.15</td>
      <td>challenge</td>
      <td>['ball_action_forced']</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1606b0e6_0</td>
      <td>200.265822</td>
      <td>start</td>
      <td>NaN</td>
      <td>1606b0e6_0</td>
      <td>202.765822</td>
      <td>end</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1606b0e6_0</td>
      <td>210.87</td>
      <td>challenge</td>
      <td>['opponent_dispossessed']</td>
      <td>5</td>
      <td>4</td>
      <td>6</td>
      <td>1606b0e6_0</td>
      <td>210.124111</td>
      <td>start</td>
      <td>NaN</td>
      <td>1606b0e6_0</td>
      <td>212.624111</td>
      <td>end</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



```python
#validate data: 
#start timestamp should be no larger than the event's timestamp
#end timestamp should be no smaller than the event's timestamp
event_df[(event_df['s_time']>event_df['time'])|(event_df['e_time']<event_df['time'])]
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
      <th>video_id</th>
      <th>time</th>
      <th>event</th>
      <th>event_attributes</th>
      <th>seq</th>
      <th>s_seq</th>
      <th>e_seq</th>
      <th>s_video_id</th>
      <th>s_time</th>
      <th>s_event</th>
      <th>s_event_attributes</th>
      <th>e_video_id</th>
      <th>e_time</th>
      <th>e_event</th>
      <th>e_event_attributes</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
#events without start timestamp
print(event_df[~event_df['s_time'].isna()].shape)
event_df[event_df['s_time'].isna()]
```

    (3418, 18)
    




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
      <th>video_id</th>
      <th>time</th>
      <th>event</th>
      <th>event_attributes</th>
      <th>seq</th>
      <th>s_seq</th>
      <th>e_seq</th>
      <th>s_video_id</th>
      <th>s_time</th>
      <th>s_event</th>
      <th>s_event_attributes</th>
      <th>e_video_id</th>
      <th>e_time</th>
      <th>e_event</th>
      <th>e_event_attributes</th>
      <th>gap_event_start</th>
      <th>gap_event_end</th>
      <th>gap_start_end</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>1606b0e6_0</td>
      <td>239.350</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>18</td>
      <td>17</td>
      <td>19</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1606b0e6_0</td>
      <td>240.401851</td>
      <td>end</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.051851</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1606b0e6_0</td>
      <td>244.590</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>22</td>
      <td>21</td>
      <td>23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1606b0e6_0</td>
      <td>246.030453</td>
      <td>end</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.440453</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1606b0e6_0</td>
      <td>253.470</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>29</td>
      <td>28</td>
      <td>30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1606b0e6_0</td>
      <td>253.990761</td>
      <td>end</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.520761</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1606b0e6_0</td>
      <td>261.310</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>36</td>
      <td>35</td>
      <td>37</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1606b0e6_0</td>
      <td>263.150</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>37</td>
      <td>36</td>
      <td>38</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1606b0e6_0</td>
      <td>265.019283</td>
      <td>end</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.869283</td>
      <td>NaN</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4360</th>
      <td>ecf251d4_0</td>
      <td>2958.587</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>924</td>
      <td>923</td>
      <td>925</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ecf251d4_0</td>
      <td>2959.156345</td>
      <td>end</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.569345</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4363</th>
      <td>ecf251d4_0</td>
      <td>2968.147</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>931</td>
      <td>930</td>
      <td>932</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ecf251d4_0</td>
      <td>2969.319076</td>
      <td>end</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.172076</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4367</th>
      <td>ecf251d4_0</td>
      <td>2997.827</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>941</td>
      <td>940</td>
      <td>942</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ecf251d4_0</td>
      <td>2998.283227</td>
      <td>end</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.456227</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4375</th>
      <td>ecf251d4_0</td>
      <td>3029.707</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>963</td>
      <td>962</td>
      <td>964</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ecf251d4_0</td>
      <td>3030.127462</td>
      <td>end</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.420462</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4379</th>
      <td>ecf251d4_0</td>
      <td>3053.067</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>973</td>
      <td>972</td>
      <td>974</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ecf251d4_0</td>
      <td>3053.744023</td>
      <td>end</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.677023</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>964 rows × 18 columns</p>
</div>




```python
#events without end timestamp
print(event_df[~event_df['e_time'].isna()].shape)
event_df[event_df['e_time'].isna()]
```

    (3418, 18)
    




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
      <th>video_id</th>
      <th>time</th>
      <th>event</th>
      <th>event_attributes</th>
      <th>seq</th>
      <th>s_seq</th>
      <th>e_seq</th>
      <th>s_video_id</th>
      <th>s_time</th>
      <th>s_event</th>
      <th>s_event_attributes</th>
      <th>e_video_id</th>
      <th>e_time</th>
      <th>e_event</th>
      <th>e_event_attributes</th>
      <th>gap_event_start</th>
      <th>gap_event_end</th>
      <th>gap_start_end</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>1606b0e6_0</td>
      <td>236.710</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>17</td>
      <td>16</td>
      <td>18</td>
      <td>1606b0e6_0</td>
      <td>236.248227</td>
      <td>start</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.461773</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1606b0e6_0</td>
      <td>242.390</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>21</td>
      <td>20</td>
      <td>22</td>
      <td>1606b0e6_0</td>
      <td>241.635933</td>
      <td>start</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.754067</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1606b0e6_0</td>
      <td>250.750</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>28</td>
      <td>27</td>
      <td>29</td>
      <td>1606b0e6_0</td>
      <td>250.223514</td>
      <td>start</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.526486</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1606b0e6_0</td>
      <td>258.830</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>35</td>
      <td>34</td>
      <td>36</td>
      <td>1606b0e6_0</td>
      <td>258.273235</td>
      <td>start</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.556765</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1606b0e6_0</td>
      <td>261.310</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>36</td>
      <td>35</td>
      <td>37</td>
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
      <td>NaN</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4359</th>
      <td>ecf251d4_0</td>
      <td>2955.027</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>923</td>
      <td>922</td>
      <td>924</td>
      <td>ecf251d4_0</td>
      <td>2954.506795</td>
      <td>start</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.520205</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4362</th>
      <td>ecf251d4_0</td>
      <td>2964.747</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>930</td>
      <td>929</td>
      <td>931</td>
      <td>ecf251d4_0</td>
      <td>2964.347000</td>
      <td>start</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.400000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4366</th>
      <td>ecf251d4_0</td>
      <td>2994.987</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>940</td>
      <td>939</td>
      <td>941</td>
      <td>ecf251d4_0</td>
      <td>2993.931590</td>
      <td>start</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.055410</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4374</th>
      <td>ecf251d4_0</td>
      <td>3026.987</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>962</td>
      <td>961</td>
      <td>963</td>
      <td>ecf251d4_0</td>
      <td>3025.405235</td>
      <td>start</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.581765</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4378</th>
      <td>ecf251d4_0</td>
      <td>3050.347</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>972</td>
      <td>971</td>
      <td>973</td>
      <td>ecf251d4_0</td>
      <td>3049.497881</td>
      <td>start</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.849119</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>964 rows × 18 columns</p>
</div>




```python
#events without start and end timestamp
event_df[(event_df['e_time'].isna()) & (event_df['s_time'].isna())]
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
      <th>video_id</th>
      <th>time</th>
      <th>event</th>
      <th>event_attributes</th>
      <th>seq</th>
      <th>s_seq</th>
      <th>e_seq</th>
      <th>s_video_id</th>
      <th>s_time</th>
      <th>s_event</th>
      <th>s_event_attributes</th>
      <th>e_video_id</th>
      <th>e_time</th>
      <th>e_event</th>
      <th>e_event_attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>1606b0e6_0</td>
      <td>261.310</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>36</td>
      <td>35</td>
      <td>37</td>
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
      <th>25</th>
      <td>1606b0e6_0</td>
      <td>298.790</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>59</td>
      <td>58</td>
      <td>60</td>
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
      <th>48</th>
      <td>1606b0e6_0</td>
      <td>454.670</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>116</td>
      <td>115</td>
      <td>117</td>
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
      <th>52</th>
      <td>1606b0e6_0</td>
      <td>480.830</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>124</td>
      <td>123</td>
      <td>125</td>
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
      <th>168</th>
      <td>1606b0e6_0</td>
      <td>1222.510</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>424</td>
      <td>423</td>
      <td>425</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4182</th>
      <td>ecf251d4_0</td>
      <td>1448.227</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>452</td>
      <td>451</td>
      <td>453</td>
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
      <th>4289</th>
      <td>ecf251d4_0</td>
      <td>2234.627</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>737</td>
      <td>736</td>
      <td>738</td>
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
      <th>4354</th>
      <td>ecf251d4_0</td>
      <td>2939.387</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>914</td>
      <td>913</td>
      <td>915</td>
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
      <th>4355</th>
      <td>ecf251d4_0</td>
      <td>2942.347</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>915</td>
      <td>914</td>
      <td>916</td>
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
      <th>4356</th>
      <td>ecf251d4_0</td>
      <td>2944.227</td>
      <td>play</td>
      <td>['pass', 'openplay']</td>
      <td>916</td>
      <td>915</td>
      <td>917</td>
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
<p>168 rows × 15 columns</p>
</div>




```python
event_df[(event_df['e_time'].isna()) & (event_df['s_time'].isna())]['event'].value_counts()
```




    play         151
    challenge     17
    Name: event, dtype: int64




```python
#the gap between start and event, event and end, and start and end
event_df['gap_event_start'] = event_df['time']- event_df['s_time']
event_df['gap_event_end'] = event_df['e_time']- event_df['time']
event_df['gap_start_end'] = event_df['e_time']- event_df['s_time']
```


```python
event_df[['gap_event_start', 'gap_event_end', 'gap_start_end']].describe()
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
      <th>gap_event_start</th>
      <th>gap_event_end</th>
      <th>gap_start_end</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3418.000000</td>
      <td>3418.000000</td>
      <td>2.622000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.122575</td>
      <td>1.278801</td>
      <td>2.500000e+00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.496239</td>
      <td>0.502026</td>
      <td>2.275300e-14</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.329385</td>
      <td>0.400928</td>
      <td>2.500000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.667994</td>
      <td>0.851162</td>
      <td>2.500000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.066239</td>
      <td>1.278637</td>
      <td>2.500000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.525130</td>
      <td>1.713679</td>
      <td>2.500000e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.099072</td>
      <td>2.163005</td>
      <td>2.500000e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
event_df[['gap_event_start', 'gap_event_end', 'gap_start_end']].hist(bins=50)
```




    array([[<AxesSubplot:title={'center':'gap_event_start'}>,
            <AxesSubplot:title={'center':'gap_event_end'}>],
           [<AxesSubplot:title={'center':'gap_start_end'}>, <AxesSubplot:>]],
          dtype=object)




    
![png](img/output_24_1.png)
    



```python
a = event_df['event'].value_counts()
a.name='cnt'
b = event_df['event'].value_counts()/event_df.shape[0]
b.name='pct'
pd.concat([a, b], axis=1)
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
      <th>cnt</th>
      <th>pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>play</th>
      <td>3586</td>
      <td>0.818348</td>
    </tr>
    <tr>
      <th>challenge</th>
      <td>624</td>
      <td>0.142401</td>
    </tr>
    <tr>
      <th>throwin</th>
      <td>172</td>
      <td>0.039251</td>
    </tr>
  </tbody>
</table>
</div>




```python
event_df['gap_start_end'].unique()
```




    array([2.5, 2.5, nan, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])



#### Explore the video data



```python
%%time
video_path = '/kaggle/input/dfl-bundesliga-data-shootout/train/1606b0e6_0.mp4'
vid = imageio.get_reader(video_path,  'ffmpeg')
fps = vid.get_meta_data()['fps']#frames per second (FPS)
print(f'frames per second (FPS): {fps}')
print('meta data of the video')
print(vid.get_meta_data())
n_frames = vid.count_frames()
print(f'number of frames: {n_frames}')
```

    frames per second (FPS): 25.0
    meta data of the video
    {'plugin': 'ffmpeg', 'nframes': inf, 'ffmpeg_version': '5.1 built with gcc 10.3.0 (conda-forge gcc 10.3.0-16)', 'codec': 'h264', 'pix_fmt': 'yuv420p(progressive)', 'fps': 25.0, 'source_size': (1920, 1080), 'size': (1920, 1080), 'rotate': 0, 'duration': 3436.6}
    number of frames: 85915
    CPU times: user 10.2 ms, sys: 27.2 ms, total: 37.5 ms
    Wall time: 1.23 s
    


```python
%%time
#display a few frames from the video

nums = [5006, 287, 5028, 5069]
for num in nums:
    image = vid.get_data(num)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    timestamp = float(num)/ fps
    plt.title(f'image #{num}, timestamp={timestamp}', fontsize=20)
    plt.show()
```


    
![png](img/output_29_0.png)
    



    
![png](img/output_29_1.png)
    



    
![png](img/output_29_2.png)
    



    
![png](img/output_29_3.png)
    


    CPU times: user 2.6 s, sys: 1.18 s, total: 3.78 s
    Wall time: 5.92 s
    


```python
for i, img in enumerate(vid):
    print('Mean of frame %i is %1.1f.' % (i, img.mean()))
    print(f'shape of the frame is {img.shape}')
    if i>5:
        break
```

    Mean of frame 0 is 76.6.
    shape of the frame is (1080, 1920, 3)
    Mean of frame 1 is 76.6.
    shape of the frame is (1080, 1920, 3)
    Mean of frame 2 is 76.6.
    shape of the frame is (1080, 1920, 3)
    Mean of frame 3 is 76.6.
    shape of the frame is (1080, 1920, 3)
    Mean of frame 4 is 76.6.
    shape of the frame is (1080, 1920, 3)
    Mean of frame 5 is 76.6.
    shape of the frame is (1080, 1920, 3)
    Mean of frame 6 is 76.6.
    shape of the frame is (1080, 1920, 3)
    


```python
#show a short clip from the video
tmp_file = f"0.mp4"
ffmpeg_extract_subclip(
    video_path, 214.23, 224.23,
    targetname=tmp_file
)
    
Video(tmp_file, width=800)
```

    Moviepy - Running:
    >>> "+ " ".join(cmd)
    Moviepy - Command successful
    




<video src="0.mp4" controls  width="800" >
      Your browser does not support the <code>video</code> element.
    </video>




```python
vid.close()
del vid
gc.collect()
```




    13002




```python

```
