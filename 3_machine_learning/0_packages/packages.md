---
title: Machine Learning Essential Packages - 2021
---
***

### Machine Learning Essential Packages - 2021

*Dec 16, 2021*


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

or, if you prefer to run pip one by one, try the following

---

	pip install lightgbm==3.3.1
	pip install scikit-learn==1.0.1
	pip install scipy==1.7.3
	pip install xgboost==1.5.1
	pip install pytorch-forecasting==0.9.2
	pip install pandas==1.3.4
	pip install hyperopt==0.1.2
	pip install mysql-connector==2.2.9
	pip install openpyxl==3.0.7
	pip install XlsxWriter==3.0.1
	pip install xlrd==2.0.1
	pip install seaborn==0.11.2
	pip install matplotlib==3.5.0
	pip install beautifulsoup4==4.10.0
	pip install Scrapy==2.5.1
	pip install plotly==5.3.1
	pip install pytorch-tabnet==3.1.1

---