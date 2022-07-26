---
title: Machine Learning Essential Packages - Jun 2022
---
***

### Machine Learning Essential Packages - Jun 2022

*Jun 30, 2022*


####  The bascis

The basic packages for data processing, mathematical and scitific calculations.


| Package | Description | 
| ------  | ------      |
| [pandas](https://pandas.pydata.org/) | data analysis and manipulation.  read/write excel/csv/compressed files, process data, and plot ugly graphs| 
| [numpy](https://numpy.org/) |  mathematical functions, random number generators, linear algebra routines. | 
| [scipy](https://scipy.org/ ) | optimization, integration, interpolation, eigenvalue problems, algebraic equations, differential equations, statistics | 

####  The core

| Package | Description | 
| ------  | ------      |
| [statsmodels](https://www.statsmodels.org/stable/index.html) | statistical models and hypothesis tests |
| [scikit-learn](https://scikit-learn.org/stable/) | this is the must-have package for any machine learning project |
| [xgboost](https://xgboost.readthedocs.io/en/stable/) | gradient boosting trees. For a long period of time, it was the Kaggle competition particpants' favorite. |
| [lightgbm](https://lightgbm.readthedocs.io/en/latest/) | also gradient boosting trees. it took xgboost's place and became the favorite of Kaggle competition particpants. |
| [pytorch](https://pytorch.org/) | I prefer it over keras and tensorflow. The cornerstone for deep learning models.|

####  data visualization

these packages help create descent looking graphs

| Package | Description | 
| ------  | ------      |
| [plotly](https://plotly.com/)| easy to use and can create some nice graphs |
| [seaborn](https://seaborn.pydata.org/) | often work hand-in-hand with matplotlib. need some skills to create presentable graphs.|
| [matplotlib](https://matplotlib.org/) | easy to create simple but ugly graphs with it. but takes real skills to create   graphs that can pass .|

####  web scraping

| Package | Description | 
| ------  | ------      |
| [Scrapy](https://scrapy.org/)| for web scraping|
| [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)| extract data from html pages|

####  nice deep learning packages based on pytorch

| Package | Description | 
| ------  | ------      |
| [pytorch-tabnet](https://github.com/dreamquark-ai/tabnet) | based on pytorch. it is becoming very popular on Kaggle|
| [pytorch-forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/) | it has the potential to be practitioners new favorite. it has some really cool deep learning algorithms.|
| [pytorch-lightning](https://www.pytorchlightning.ai/) | if you install pytorch-forecasting, this package along with pytorch will also be installed|

####  hyperparameter tunning

either one of the following is sufficient to do the job

| Package | Description | 
| ------  | ------      |
| [hyperopt](http://hyperopt.github.io/hyperopt/) | I use it for all my projects. Very powerful package for hyperparameter tunning |
| [optuna](https://optuna.org/) | an alertnative for hyperopt. I don't use it but it is used in pytorch-forecasting package and gets installed when installing pytorch-forecasting|

####  feature engineering

| Package | Description | 
| ------  | ------      |
| [ta-lib](https://github.com/mrjbq7/ta-lib) | This is a popular package for engineering technical features for time series data. Note that this package requires *Visual Studio Community 2015* be installed on the machine first|
| [finta](https://github.com/peerchemist/finta) | This package is a nice alternative if installing ta-lib is not possible due to various reasons |
| [tsfresh](https://tsfresh.readthedocs.io/en/latest/text/introduction.html) | tsfresh is used for systematic feature engineering from time-series and other sequential data|

####  anomaly detection

| Package | Description | 
| ------  | ------      |
| [Merlion](https://github.com/salesforce/Merlion) | This is a descent package for anomaly detection. It is developed by Salesforce.|


#### misc

| Package | Description | 
| ------  | ------      |
| [yfinance](https://pypi.org/project/yfinance/) | This is a very neat package that helps downloading stock price data from yahoo finance.|
| [PyArrow](https://arrow.apache.org/docs/python/index.html) | I use this package to write Parquet file.|


#### more....
- the following is a list of packages that includes all the ones mentioned above, plus a few others. 
- copy the list to a 'requirements.txt' and run pip on the txt file can get all of them installed at once.

---

	joblib==1.1.0
	numpy==1.21.4
	scipy==1.7.3
	pandas==1.3.4
	scikit-learn==1.0.1
	lightgbm==3.3.1
	xgboost==1.5.1
	tsfresh==0.17.0
	pytorch-forecasting==0.9.2
	pytorch-lightning==1.5.6
	hyperopt==0.1.2
	mysql-connector==2.2.9
	openpyxl==3.0.7
	XlsxWriter==3.0.1
	xlrd==2.0.1
	seaborn==0.11.2
	statsmodels==0.13.1
	beautifulsoup4==4.10.0
	Scrapy==2.5.1
	plotly==5.3.1
	matplotlib==3.5.0
	pytorch-tabnet==3.1.1
	optuna==2.10.0 

---

or, if you prefer to run pip one by one, try the following.

---

	pip install pandas==1.3.4
	pip install scipy==1.7.3
	pip install numpy==1.21.4
	pip install scikit-learn==1.0.1

	python -m pip install --upgrade pip

	pip install lightgbm==2.3.1 
	pip install xgboost==1.5.1
	pip install hyperopt==0.1.2
	pip install seaborn==0.11.2
	pip install matplotlib==3.5.0
	pip install plotly==5.3.1

	pip install pytorch-tabnet==3.1.1
	pip install pytorch-forecasting==0.9.2
	pip install pytorch-lightning==1.5.5


	pip install openpyxl==3.0.7
	pip install XlsxWriter==3.0.1
	pip install xlrd==2.0.1
	pip install mysql-connector==2.2.9

	pip install beautifulsoup4==4.10.0
	pip install Scrapy==2.5.1 
	pip install notebook==6.4.6

	pip3 install torch torchvision torchaudio

	pip install tsfresh==0.17.0
	pip install statsmodels==0.10.2

	pip install yfinance==0.1.66
	pip install pyarrow==6.0.1
	pip install TA-Lib==0.4.23
	pip install h5py==3.7.0

---

**A few more notes**

- Tabnet wheel file can be downloaded from here:
 > https://files.pythonhosted.org/packages/94/e5/2a808d611a5d44e3c997c0d07362c04a56c70002208e00aec9eee3d923b5/pytorch_tabnet-3.1.1-py3-none-any.whl
- You may need to upgrade pip first in order to isntall *Scrapy* successfully. 
- tsfresh: *tsfresh==0.17.0* requires "statsmodels==0.10.2", higher version of statsmodels will lead to error. 
- Microsoft Visual C++ Redistributable
> - DLL load failure may occur if it is not installed.
> - It can be downloaded at https://aka.ms/vs/16/release/vc_redist.x64.exe
