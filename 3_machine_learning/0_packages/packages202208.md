---
title: Machine Learning Essential Packages - Aug 2022
---
***

### Machine Learning Essential Packages - Aug 2022

*Aug 6, 2022*

`*` indicates packages newly added to the lists.

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

| Package | Description | 
| ------  | ------      |
| [plotly](https://plotly.com/)| easy to use and can create some nice graphs |
| [seaborn](https://seaborn.pydata.org/) | often work hand-in-hand with matplotlib. need some skills to create presentable graphs.|
| [matplotlib](https://matplotlib.org/) | easy to create simple but ugly graphs with it. but takes real skills to create   graphs that can pass .|
| *[mplfinance](https://github.com/matplotlib/mplfinance) | built on top of matplotlib, this package specialize in visualizing financial time series data. |
| *[networkx](https://networkx.org/) | network analysis and visualization |

####  web scraping

| Package | Description | 
| ------  | ------      |
| [Scrapy](https://scrapy.org/)| for web scraping|
| [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)| extract data from html pages|

####  deep learning packages based on pytorch

| Package | Description | 
| ------  | ------      |
| [pytorch-tabnet](https://github.com/dreamquark-ai/tabnet) | based on pytorch. it is becoming very popular on Kaggle|
| [pytorch-forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/) | it has the potential to be practitioners new favorite. it has some really cool deep learning algorithms.|
| [pytorch-lightning](https://www.pytorchlightning.ai/) | if you install pytorch-forecasting, this package along with pytorch will also be installed|

####  hyperparameter tunning

| Package | Description | 
| ------  | ------      |
| [hyperopt](http://hyperopt.github.io/hyperopt/) | Very powerful package for hyperparameter tunning |
| [optuna](https://optuna.org/) | an alertnative to hyperopt. it is used in pytorch-forecasting package and gets installed when installing pytorch-forecasting|

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
| *[STUMPY](https://github.com/TDAmeritrade/stumpy) | This package is developed by TD Ameritrade.|

#### data processing and big data

| Package | Description | 
| ------  | ------      |
|*[dask](https://www.dask.org/) | helps handle big data files.|
|*[Pillow](https://pillow.readthedocs.io/en/stable/reference/Image.html) | Image data processing and manipulation.|
|*[PyArrow](https://arrow.apache.org/docs/python/index.html) |Feather and Parquet file.|
| [XlsxWriter](https://xlsxwriter.readthedocs.io/) | required package to pandas to write Excel files.|
| [xlrd](https://xlrd.readthedocs.io/en/latest/) | for reading and writing Excel files.|
|*[PyYAML](https://pyyaml.org/) | read and write YAML file.|
|*[h5py](https://docs.h5py.org/en/stable/) | The h5py package is a Pythonic interface to the HDF5 binary data format.|

#### misc

| Package | Description | 
| ------  | ------      |
| [yfinance](https://pypi.org/project/yfinance/) | This is a very neat package that helps downloading stock price data from yahoo finance.|
| [mysql-connector](https://pypi.org/project/mysql-connector/) | to connect, read, and write mysql database.|
|*[numba](https://numba.pydata.org/) | Numba is an open source JIT compiler that translates a subset of Python and NumPy code into fast machine code.|
|*[shap](https://shap.readthedocs.io/en/latest/index.html) | SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model.|
|*[dtreeviz](https://github.com/parrt/dtreeviz) | A python library for decision tree visualization and model interpretation.|


#### more....
- the following is a list of packages that includes all the ones mentioned above, plus a few others. 
- copy the list to a 'requirements.txt' and run pip on the txt file can get all of them installed at once.

---

	numpy==1.21.4
	pandas==1.3.4
	scipy==1.7.3

	statsmodels==0.10.2
	scikit-learn==1.0.1
	xgboost==1.5.1
	lightgbm==3.3.2
	torch==1.12.0
	torchaudio==0.11.0
	torchaudio==0.12.0
	torchmetrics==0.9.1
	torchvision==0.12.0
	torchvision==0.13.0
	torchviz==0.0.2

	plotly==5.3.1
	seaborn==0.11.2
	matplotlib==3.5.0
	matplotlib-inline==0.1.3
	mplfinance==0.12.9b1
	networkx==2.6.3

	Scrapy==2.5.1
	beautifulsoup4==4.10.0

	pytorch-forecasting==0.9.2
	pytorch-lightning==1.5.5
	pytorch-tabnet==3.1.1

	hyperopt==0.1.2
	optuna==2.10.0

	TA-Lib==0.4.23
	tsfresh==0.17.0

	dask==2022.2.0
	Pillow==9.1.1
	pyarrow==6.0.1
	xlrd==2.0.1
	XlsxWriter==3.0.1
	openpyxl==3.0.7
	PyYAML==6.0
	pymongo==4.1.1
	h5py==3.7.0

	yfinance==0.1.66
	mysql-connector==2.2.9
	numba==0.55.2
	tqdm==4.64.0

	shap==0.41.0
	dtreeviz==1.3.7
	graphviz==0.20.1

	jedi==0.18.1
	Jinja2==3.1.2
	joblib==1.1.0
	notebook==6.4.6

---

or, if you prefer to run pip one by one, try the following.

---

	pip install numpy==1.21.4
	pip install pandas==1.3.4
	pip install scipy==1.7.3

	pip install statsmodels==0.10.2
	pip install scikit-learn==1.0.1
	pip install xgboost==1.5.1
	pip install lightgbm==3.3.2
	pip install torch==1.12.0
	pip install torchaudio==0.11.0
	pip install torchaudio==0.12.0
	pip install torchmetrics==0.9.1
	pip install torchvision==0.12.0
	pip install torchvision==0.13.0
	pip install torchviz==0.0.2

	pip install plotly==5.3.1
	pip install seaborn==0.11.2
	pip install matplotlib==3.5.0
	pip install matplotlib-inline==0.1.3
	pip install mplfinance==0.12.9b1
	pip install networkx==2.6.3

	pip install Scrapy==2.5.1
	pip install beautifulsoup4==4.10.0

	pip install pytorch-forecasting==0.9.2
	pip install pytorch-lightning==1.5.5
	pip install pytorch-tabnet==3.1.1

	pip install hyperopt==0.1.2
	pip install optuna==2.10.0

	pip install TA-Lib==0.4.23
	pip install tsfresh==0.17.0

	pip install dask==2022.2.0
	pip install Pillow==9.1.1
	pip install pyarrow==6.0.1
	pip install xlrd==2.0.1
	pip install XlsxWriter==3.0.1
	pip install openpyxl==3.0.7
	pip install PyYAML==6.0
	pip install pymongo==4.1.1
	pip install h5py==3.7.0

	pip install yfinance==0.1.66
	pip install mysql-connector==2.2.9
	pip install numba==0.55.2
	pip install tqdm==4.64.0

	pip install shap==0.41.0
	pip install dtreeviz==1.3.7
	pip install graphviz==0.20.1

	pip install jedi==0.18.1
	pip install Jinja2==3.1.2
	pip install joblib==1.1.0
	pip install notebook==6.4.6


---

**A few more notes**

- Tabnet wheel file can be downloaded from here:
 > https://files.pythonhosted.org/packages/94/e5/2a808d611a5d44e3c997c0d07362c04a56c70002208e00aec9eee3d923b5/pytorch_tabnet-3.1.1-py3-none-any.whl
- You may need to upgrade pip first in order to isntall *Scrapy* successfully. 
- tsfresh: *tsfresh==0.17.0* requires "statsmodels==0.10.2", higher version of statsmodels will lead to error. 
- Microsoft Visual C++ Redistributable
> - DLL load failure may occur if it is not installed.
> - It can be downloaded at https://aka.ms/vs/16/release/vc_redist.x64.exe
