---
layout: default
title: Installing Merlion
parent: Time Series Anomaly Detection
grand_parent: Time Series
nav_order: 1
---

## Notes on installing Merlion


- reference: [github](https://github.com/salesforce/Merlion)
- according to the github page,using `pip install salesforce-merlion` should be sufficient to have the Merlion package installed. 
- however, on Windows machine, an error can occur due to Merlion's dependency package `fbprophet`. 
- in order to have merlion package be installed successfully, we need to first install `fbprophet` package. This [stack overflow](https://stackoverflow.com/questions/53178281/installing-fbprophet-python-on-windows-10) page provides useful tricks to fix issues with installing `fbprophet` package on Windows machine.
- what did not work for me: first run `pip install pystan==2.18.0.0`, then run `pip install fbprophet`. 
- what worked for me: 
   - first run `pip install pystan==2.17.1.0`. This step will uninstall whatever version of pystan package on the machine and isntall the version specified in the pip command.
   - then run `pip install fbprophet`. This step will retrieve the latest pystan version, uninstall the version installed from previous step and install the latest version. The successfuly installation message `Successfully installed cmdstanpy-0.9.68 prophet-1.0.1 pystan-2.19.1.1`.
   