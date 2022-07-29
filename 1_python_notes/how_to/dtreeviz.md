### How to install dtreeviz for Decision Tree Visualization on Windows without system admin permissions

*Jul 29, 2022*


references:
- [https://github.com/parrt/dtreeviz](https://github.com/parrt/dtreeviz)
- [add system path in jupyter notebook](https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft)

steps: 
1. open cmd window and run the following scripts
```python
pip install dtreeviz             # install dtreeviz for sklearn
pip install dtreeviz[xgboost]    # install XGBoost related dependency
pip install dtreeviz[pyspark]    # install pyspark related dependency
pip install dtreeviz[lightgbm]   # install LightGBM related dependency
```
2. download the graphviz file
    - search `2.38.msi` on this page [https://www2.graphviz.org/Archive/stable/windows/](https://www2.graphviz.org/Archive/stable/windows/)
    - download the file to local and unzip it

3. in jupyter notebook, run the following lines
    - replace `C:/Users/abc/Documents/Graphviz` with the actual directory of `Graphviz` folder
```python
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/abc/Documents/Graphviz/bin;C:/Users/abc/Documents/Graphviz/bin/dot.exe;'
```
