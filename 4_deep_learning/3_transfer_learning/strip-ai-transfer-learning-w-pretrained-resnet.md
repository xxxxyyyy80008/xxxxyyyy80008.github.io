---
title: Mayo Clinic - STRIP AI - Transfer Learning with pytorch pretrained resnet
---

***

## Mayo Clinic - STRIP AI - Transfer Learning with pytorch pretrained resnet

*Jul 28, 2022*

#### Key notes:
- this script is based on [pytorch transfer learning example](https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py) using pretrained resnet
- dataset used in this notebook are images resized to max(height,width)=500 using `PIL` (pillow) package
    -  note this dataset is missing 2 images - one from *train* folder and one from *other* folder
    -  for EDA and image processing, refer [this notebook](https://www.kaggle.com/code/xxxxyyyy80008/mayo-clinic-strip-ai-eda-and-image-processing) and this post [Process images with pyvips package and handle the memory limitation issue](https://www.kaggle.com/competitions/mayo-clinic-strip-ai/discussion/340052)
- the pytorch pretrained models are downloaded from pytorch website and added to the notebook via `+Add Data`
    -  this is to make the script still work when the notebook is set `offline`
- for simplicity, there is no addtional image processing other than resizing the original images



#### References: 
- [pytorch transfer learning totorial](https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py)
- [pytorch list of pretrained models](https://pytorch.org/vision/stable/models.html)


```python
import os
next(os.walk('/kaggle/input'))   
```




    ('/kaggle/input',
     ['stripai-traindata', 'pytorch-pretrained', 'mayo-clinic-strip-ai'],
     [])




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
import shutil
#additional data processing

import pyarrow.parquet as pq
import pyarrow as pa

from sklearn.preprocessing import StandardScaler, MinMaxScaler


#visualization
import seaborn as sns
import matplotlib.pyplot as plt

#load images
import matplotlib.image as mpimg
import PIL
from PIL import Image




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




```python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms


cudnn.benchmark = True
plt.ion()   # interactive mode
```




    <matplotlib.pyplot._IonContext at 0x7fd5f7066490>



#### Load data and prep data


```python
train_df = pd.read_csv('/kaggle/input/mayo-clinic-strip-ai/train.csv')
other_df = pd.read_csv('/kaggle/input/mayo-clinic-strip-ai/other.csv')
```


```python
other_df['label'] = 'Other'
```


```python
df = pd.concat([train_df[['image_id', 'patient_id', 'label']], 
                other_df[['image_id', 'patient_id', 'label']]], axis=0)
df.shape
```




    (1150, 3)




```python
del train_df, other_df
gc.collect()
```




    151




```python
#check the distribution of target labels
df['label'].value_counts()/df.shape[0]
```




    CE       0.475652
    Other    0.344348
    LAA      0.180000
    Name: label, dtype: float64




```python
#convert the lable into numeric type
label_dict ={'CE':0, 'LAA':1, 'Other':2}
df['target']=df['label'].map(label_dict)
df['target'].value_counts()/df.shape[0]
```




    0    0.475652
    2    0.344348
    1    0.180000
    Name: target, dtype: float64




```python
from sklearn.model_selection import train_test_split
```


```python
#split data into train and eval sets
X_train, X_val, y_train, y_val = train_test_split(df[['image_id', 'label', 'target']], df['target'], test_size=0.25, random_state=1234)
```


```python
#check the label distribution in train and val datasets
print('train data: ', len(X_train),  '-'*50)
display(X_train['label'].value_counts()/X_train.shape[0])
print('valuation data:', len(X_val), '-'*50)
display(X_val['label'].value_counts()/X_val.shape[0])
```

    train data:  862 --------------------------------------------------
    


    CE       0.476798
    Other    0.345708
    LAA      0.177494
    Name: label, dtype: float64


    valuation data: 288 --------------------------------------------------
    


    CE       0.472222
    Other    0.340278
    LAA      0.187500
    Name: label, dtype: float64


### Define Dataset class


```python
#define image dataset class
import torch
from torch.utils.data import (Dataset, DataLoader)



img_folder = '/kaggle/input/stripai-traindata/train_images/all'


class IMG_Data(Dataset):
    
    def __init__(self, data): 
        

        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        image_id =self.data.iloc[idx]['image_id']
        img_path = f"({img_folder}/{image_id}.tif"
        y = self.data.iloc[idx]['target']
        if Path(img_path).exists():
            img = Image.open(img_path)
            #transform image: crop image >> to tensor >> normalize
            img = transforms.functional.center_crop(img, 500)
            img = np.asarray(img, np.uint8)            
        else:
            img = np.zeros((500, 500, 3), np.uint8)
            
        x = torch.FloatTensor(img.transpose((2,0,1))  ) #need to slip the data in shape n_channels*height*width
        
        return x, y
    

def load_data(df, batch_size, n_workers=0, shuffle=False):
    data = IMG_Data(df)
    
    loader = DataLoader(data, batch_size=batch_size, num_workers=n_workers, shuffle=shuffle)
    
    return loader
```


```python
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

```

### Train model - resnet18: train the last layer

The following model training process will freeze all the network except the final layer.



```python
pl.seed_everything(random_seed)
```




    1234




```python
%%time
#load data to dataloaer
dataloaders = {'train': load_data(X_train, 4), 'val': load_data(X_val, 4)}
dataset_sizes = {'train': len(X_train), 'val':len(X_val)}
```

    CPU times: user 115 µs, sys: 18 µs, total: 133 µs
    Wall time: 137 µs
    


```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
```




    device(type='cuda', index=0)




```python

######################################################################
# ConvNet as fixed feature extractor
# ----------------------------------
#
# Here, we need to freeze all the network except the final layer. We need
# to set ``requires_grad = False`` to freeze the parameters so that the
# gradients are not computed in ``backward()``.
#
# You can read more about this in the documentation
# `here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
#

model_conv = models.resnet18()
model_conv.load_state_dict(torch.load('/kaggle/input/pytorch-pretrained/resnet18-f37072fd.pth'))
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 3)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()


# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
```


```python
%%time
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=50)
```

    Epoch 0/49
    ----------
    train Loss: 1.2066 Acc: 0.4037
    val Loss: 1.1667 Acc: 0.1875
    
    Epoch 1/49
    ----------
    train Loss: 1.2094 Acc: 0.3910
    val Loss: 1.1667 Acc: 0.1875
    
    Epoch 2/49
    ----------
    train Loss: 1.2094 Acc: 0.3910
    val Loss: 1.1667 Acc: 0.1875
    
    Epoch 3/49
    ----------
    train Loss: 1.2094 Acc: 0.3910
    val Loss: 1.1667 Acc: 0.1875
    
    Epoch 4/49
    ----------
    train Loss: 1.2094 Acc: 0.3910
    val Loss: 1.1667 Acc: 0.1875
    
    Epoch 5/49
    ----------
    train Loss: 1.2094 Acc: 0.3910
    val Loss: 1.1667 Acc: 0.1875
    
    Epoch 6/49
    ----------
    train Loss: 1.2094 Acc: 0.3910
    val Loss: 1.1667 Acc: 0.1875
    
    Epoch 7/49
    ----------
    train Loss: 1.0480 Acc: 0.4687
    val Loss: 1.0371 Acc: 0.4722
    
    Epoch 8/49
    ----------
    train Loss: 1.0393 Acc: 0.4687
    val Loss: 1.0371 Acc: 0.4722
    
    Epoch 9/49
    ----------
    train Loss: 1.0393 Acc: 0.4687
    val Loss: 1.0371 Acc: 0.4722
    
    Epoch 10/49
    ----------
    train Loss: 1.0393 Acc: 0.4687
    val Loss: 1.0371 Acc: 0.4722
    
    Epoch 11/49
    ----------
    train Loss: 1.0393 Acc: 0.4687
    val Loss: 1.0371 Acc: 0.4722
    
    Epoch 12/49
    ----------
    train Loss: 1.0393 Acc: 0.4687
    val Loss: 1.0371 Acc: 0.4722
    
    Epoch 13/49
    ----------
    train Loss: 1.0393 Acc: 0.4687
    val Loss: 1.0371 Acc: 0.4722
    
    Epoch 14/49
    ----------
    train Loss: 1.0292 Acc: 0.4768
    val Loss: 1.0360 Acc: 0.4722
    
    Epoch 15/49
    ----------
    train Loss: 1.0293 Acc: 0.4768
    val Loss: 1.0360 Acc: 0.4722
    
    Epoch 16/49
    ----------
    train Loss: 1.0293 Acc: 0.4768
    val Loss: 1.0359 Acc: 0.4722
    
    Epoch 17/49
    ----------
    train Loss: 1.0293 Acc: 0.4768
    val Loss: 1.0359 Acc: 0.4722
    
    Epoch 18/49
    ----------
    train Loss: 1.0293 Acc: 0.4768
    val Loss: 1.0359 Acc: 0.4722
    
    Epoch 19/49
    ----------
    train Loss: 1.0293 Acc: 0.4768
    val Loss: 1.0359 Acc: 0.4722
    
    Epoch 20/49
    ----------
    train Loss: 1.0293 Acc: 0.4768
    val Loss: 1.0359 Acc: 0.4722
    
    Epoch 21/49
    ----------
    train Loss: 1.0276 Acc: 0.4768
    val Loss: 1.0358 Acc: 0.4722
    
    Epoch 22/49
    ----------
    train Loss: 1.0275 Acc: 0.4768
    val Loss: 1.0357 Acc: 0.4722
    
    Epoch 23/49
    ----------
    train Loss: 1.0275 Acc: 0.4768
    val Loss: 1.0357 Acc: 0.4722
    
    Epoch 24/49
    ----------
    train Loss: 1.0275 Acc: 0.4768
    val Loss: 1.0356 Acc: 0.4722
    
    Epoch 25/49
    ----------
    train Loss: 1.0275 Acc: 0.4768
    val Loss: 1.0356 Acc: 0.4722
    
    Epoch 26/49
    ----------
    train Loss: 1.0275 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 27/49
    ----------
    train Loss: 1.0274 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 28/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 29/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 30/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 31/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 32/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 33/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 34/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 35/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 36/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 37/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 38/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 39/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 40/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 41/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 42/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 43/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 44/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 45/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 46/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 47/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 48/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 49/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Training complete in 5m 3s
    Best val Acc: 0.472222
    CPU times: user 4min 54s, sys: 2.03 s, total: 4min 56s
    Wall time: 5min 2s
    

### Train model - resnet50: train all layers

The following model training process will train all layers



```python
model_ft = models.resnet50()
model_ft.load_state_dict(torch.load('/kaggle/input/pytorch-pretrained/resnet50-0676ba61.pth'))
num_ftrs = model_ft.fc.in_features


model_ft.fc = nn.Linear(num_ftrs, 3)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50)
```

    Epoch 0/49
    ----------
    train Loss: 1.1166 Acc: 0.4060
    val Loss: 1.0493 Acc: 0.4722
    
    Epoch 1/49
    ----------
    train Loss: 1.0540 Acc: 0.4443
    val Loss: 1.0503 Acc: 0.4722
    
    Epoch 2/49
    ----------
    train Loss: 1.0462 Acc: 0.4559
    val Loss: 1.0408 Acc: 0.4722
    
    Epoch 3/49
    ----------
    train Loss: 1.0432 Acc: 0.4548
    val Loss: 1.0372 Acc: 0.4722
    
    Epoch 4/49
    ----------
    train Loss: 1.0408 Acc: 0.4652
    val Loss: 1.0366 Acc: 0.4722
    
    Epoch 5/49
    ----------
    train Loss: 1.0393 Acc: 0.4698
    val Loss: 1.0374 Acc: 0.4722
    
    Epoch 6/49
    ----------
    train Loss: 1.0385 Acc: 0.4710
    val Loss: 1.0378 Acc: 0.4722
    
    Epoch 7/49
    ----------
    train Loss: 1.0291 Acc: 0.4768
    val Loss: 1.0360 Acc: 0.4722
    
    Epoch 8/49
    ----------
    train Loss: 1.0291 Acc: 0.4768
    val Loss: 1.0359 Acc: 0.4722
    
    Epoch 9/49
    ----------
    train Loss: 1.0291 Acc: 0.4768
    val Loss: 1.0358 Acc: 0.4722
    
    Epoch 10/49
    ----------
    train Loss: 1.0291 Acc: 0.4768
    val Loss: 1.0358 Acc: 0.4722
    
    Epoch 11/49
    ----------
    train Loss: 1.0291 Acc: 0.4768
    val Loss: 1.0358 Acc: 0.4722
    
    Epoch 12/49
    ----------
    train Loss: 1.0291 Acc: 0.4768
    val Loss: 1.0358 Acc: 0.4722
    
    Epoch 13/49
    ----------
    train Loss: 1.0291 Acc: 0.4768
    val Loss: 1.0358 Acc: 0.4722
    
    Epoch 14/49
    ----------
    train Loss: 1.0275 Acc: 0.4768
    val Loss: 1.0357 Acc: 0.4722
    
    Epoch 15/49
    ----------
    train Loss: 1.0275 Acc: 0.4768
    val Loss: 1.0357 Acc: 0.4722
    
    Epoch 16/49
    ----------
    train Loss: 1.0275 Acc: 0.4768
    val Loss: 1.0356 Acc: 0.4722
    
    Epoch 17/49
    ----------
    train Loss: 1.0274 Acc: 0.4768
    val Loss: 1.0356 Acc: 0.4722
    
    Epoch 18/49
    ----------
    train Loss: 1.0274 Acc: 0.4768
    val Loss: 1.0356 Acc: 0.4722
    
    Epoch 19/49
    ----------
    train Loss: 1.0274 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 20/49
    ----------
    train Loss: 1.0274 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 21/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 22/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 23/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 24/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 25/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 26/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 27/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 28/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 29/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 30/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 31/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 32/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 33/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 34/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 35/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 36/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 37/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 38/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 39/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 40/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 41/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 42/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 43/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 44/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 45/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 46/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 47/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 48/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Epoch 49/49
    ----------
    train Loss: 1.0272 Acc: 0.4768
    val Loss: 1.0355 Acc: 0.4722
    
    Training complete in 25m 5s
    Best val Acc: 0.472222
    


```python

```
