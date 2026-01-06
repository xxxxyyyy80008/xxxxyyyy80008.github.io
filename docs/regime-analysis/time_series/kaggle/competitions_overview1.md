---
layout: default
title: Kaggle Time Series Competitions and Solutions (1)
parent: Kaggle Competitions
grand_parent: Time Series
nav_order: 1
---

## Kaggle Time Series Competitions and Solutions (1)

*Jul 14, 2020*


1. M5 Forecasting - Accuracy: https://www.kaggle.com/c/m5-forecasting-accuracy
1. Home Credit Default Risk: https://www.kaggle.com/c/home-credit-default-risk

<h4> Notes:</h4>

- learning API: 
    - LightGBM is becoming more popular than XGBoost. LightGBM is much faster than XGBoost and can achieve better performance in same amount of time in general.
    - Neural networks work but do not always perform better than tree-based models. (*in my experience, with limited time and computing resources, LightGBM always outperform neural networks)
    
- objective function/loss function
    - custom objective/loss function sometime helps
    
- k-fold and validation set
    - for time series data, it is extremely difficult to create a sound/robust validation set

<h3>M5 Forecasting - Accuracy </h3>

1. Kaggle URL: https://www.kaggle.com/c/m5-forecasting-accuracy
1. Top place solutions:
>  1. 1st place solution: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163684
>       - LightGBM, Stacking, time series split folds
>  1. 2nd place solution: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/164599
>       - LightGBM
>       - custom objective function
>  1. 3rd place solution - NN approach: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/164374
>       - We trained modified DeepAR with Tweedie loss and make a final prediction from the ensemble of multiple models chosen using the past 14 periods of WRMSSEs.
>       - DeepAR(https://arxiv.org/abs/1704.04110) 
>       - Tweedie Loss: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/150614
>       - GluonTS: https://github.com/zalandoresearch/pytorch-ts
>  1. 4th place solution: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163216
>       - LightGBM; objective = tweedie;
>       - no early stopping; time series split folds
>  1. 5th place solution: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163916
>       - LightGBM with early stopping
>       - Poisson objective
>  1. 14th place solution: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163211
>       - LightGBM; objective = tweedie;
>  1. 68th place: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163578
>       - 4 folds for validation
>       - single LightGBM with recursive one step ahead prediction trained over the entire dataset as base model.
>       - objective: tweedie
>       - For ensembling I simply aggregated all the predictions using the trimean, which is a robust central tendency statistic.  I decided not to use a more complex aggregation strategy after reading this thread: 
>          - Ensemble time series model: https://stats.stackexchange.com/questions/47950/ensemble-time-series-model
>          - Trimean: https://en.wikipedia.org/wiki/Trimean
>  1. 109th place: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163379
>       - Catboost; time series split folds.
>       - code: https://www.kaggle.com/altprof/109th-place-code-only
>  1. 178th Place: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163372
>       - 3-fold time series cross validation
>       - regression + classification *essentially treat classification (binary) probilities as weight for regression predictions
>  1. 219th: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163206
>       - LightGBM + XGBoost + CatBoost + NN + others
>       - objective: tweedie
>       - ensemble
>       - expanding window time series split

1. Discussions
>  1. Python NN Full Training Pipeline: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/152767
>       - code: https://github.com/Trigram19/m5-python-starter
>  1. Lightgbm: Custom Loss Anything using PyTorch Autograd!: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/152837
>  1. Simple LGBM GroupKFold CV: https://www.kaggle.com/ragnar123/simple-lgbm-groupkfold-cv
>       - LightGBM, GroupKFold


<h3>Time series resources and methodologies </h3>


1. Recursive, direct, and hybrid modeling: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/151927
>  - 4 Strategies for Multi-Step Time Series Forecasting: https://machinelearningmastery.com/multi-step-time-series-forecasting/
>      - Recursive and direct multi-step forecasting: the best of both worlds: https://robjhyndman.com/papers/rectify.pdf
>      - Recursive models have the problem of bias, but they are nice because the model has the same underlying structure for every prediction. 
Direct models are different from day to day, and can be very different, leading to high variance in the predictions.
>      - So it seems to me that we should use recursive modeling to get some precise predictions, although they are biased, and then use direct modeling to reduce the bias in our predictions. 
>          - https://cdn-images-1.medium.com/max/600/1*k_D4-U7c3Tf8hJRpaOZoBQ.png
>          - https://files.ai-pool.com/a/c457c503205d0740d3efc553bdb74b0b.png
>          - https://miro.medium.com/max/1660/1*9hPX9pAO3jqLrzt0IE3JzA.png
>  - Direct Multi-step Forecast Strategy: 
>      - prediction(t+1) = model1(obs(t-1), obs(t-2), ..., obs(t-n))
>      - prediction(t+2) = model2(obs(t-2), obs(t-3), ..., obs(t-n))
>  - Recursive Multi-step Forecast: 
>      - prediction(t+1) = model(obs(t-1), obs(t-2), ..., obs(t-n))
>      - prediction(t+2) = model(prediction(t+1), obs(t-1), ..., obs(t-n))
>  - Direct-Recursive Hybrid Strategies:
>      - prediction(t+1) = model1(obs(t-1), obs(t-2), ..., obs(t-n))
>      - prediction(t+2) = model2(prediction(t+1), obs(t-1), ..., obs(t-n))
>  - Multiple Output Strategy:
>      - prediction(t+1), prediction(t+2) = model(obs(t-1), obs(t-2), ..., obs(t-n))
1. Resources for Time Series Methods: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133476
>  - Rob Hyndman & George Athanasopoulos - Forecasting: Principles and Practice. The book can be found here
>      - https://otexts.com/fpp2/.
>      - https://otexts.com/fpp3/
>      - https://github.com/tidyverts
>  - Shumway & Stoffer - Time Series Analysis and Its Applications. The book can be found here:
>      - http://db.ucsd.edu/static/TimeSeries.pdf.
>  - Croston's method in R: https://github.com/Mcompetitions/M5-methods/blob/master/validation/Point%20Forecasts%20-%20Benchmarks.R.
>  - Croston's method in Python (my implementation): https://www.kaggle.com/robertburbidge/statistical-benchmarks-wrmsse-stochastic-ensemble.
1. Validation strategies: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133904
>  - it is hard to find/build a good validation set
>      - K-folder
>      - Group k-folder (group by storeid or deptid)
>      - Strategy k-folder (group by storeid or deptid)
>      - Hold-out (the last 28 days in train)
>      - Hold-out (the same 28 days of validation in each year)
>      - Time series k-folder

1. Time Series Cross-Validation: evolution, xgboost, features: https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/discussion/46602
>  -  A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction: https://robjhyndman.com/papers/cv-wp.pdf
 the use of standard K-fold CV is possible as long as the models considered have uncorrelated errors
    When purely (non-linear, nonparametric) autoregressive methods are applied to forecasting problems, as is often the case
(e.g., when using Machine Learning methods), the aforementioned problems of CV are largely irrelevant, and CV can and should be used without modification, as in the independent case.

<h3>Home Credit Default Risk </h3>

1. top solutions
  - 1st Place Solution: https://www.kaggle.com/c/home-credit-default-risk/discussion/64821
     - https://www.kaggle.com/c/home-credit-default-risk/discussion/64480
  - 2nd place solution: https://www.kaggle.com/c/home-credit-default-risk/discussion/64722
  - 3rd place solution: https://www.kaggle.com/c/home-credit-default-risk/discussion/64596

  - 8th Solution Overview: https://www.kaggle.com/c/home-credit-default-risk/discussion/64474
     - Our single models are all lgb models
     - key for our team is that everyone brings a good model that is different from others, key for our team is that everyone brings a good model that is different from others
     - The final results show that golden rule still holds: trust your local CV. 
     - We are using stratified 10-fold as the cross validation method. 


1. A few notes: https://www.kaggle.com/c/home-credit-default-risk/discussion/58332

   Try different K-fold sets to see if your model is stable, ..., so TRUST YOUR LOCAL CV!!!
   LGB and XGB have a rich toolset to remove noisy features and regularize your models. Two of the most important for this competition are *feature fraction* and *reglambda*.

   **tuning**:

     If you're using xgboost, switch to LightGBM, which is much faster. (That's not to say that you shouldn't eventually use XGB to build a model, but it makes more sense to start with LGB and then move on to XGB when you can no longer improve your LGB model.)

    Start with a relatively high learning rate (e.g., 0.2 or 0.1) and tune the other parameters with this high rate. Then lower the learning rate and fine-tune the parameters.

   Tune manually instead of using an automatic optimization method. I know this is old-school, but I find tuning manually gives me insights into the data that I would miss with an automatic method.

   If you are keeping the learning rate constant, you don't need to wait for early stopping to end a validation round. It's often clear much sooner that whatever tuning change you just made is making the model worse, so you can abort that validation run.

   In concert with point #4, pay attention to the training error as well as the validation error. What you want when tuning parameters is a lower validation error given a fixed training error.
