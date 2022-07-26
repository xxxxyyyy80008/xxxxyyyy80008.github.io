---
title: Imbalanced Binary Classification
---

<h2> imbalanced binary classification </h2>

<h3> a few notes </h3>

- one class accounts for only 20%, 25%, 20%, 10%, 8% of data (the type of data I've dealt with).
- based on the posts and academic papers I've read so far, there does not seem to be one single metric that does the magic.
- sometimes one post (or one's experiment results or experiences) contradicts another's.
- so it all boils down to doing your own experiments on the data you are working with.


<h4> inventory of evaluation metrics </h4>

[This article](https://www.sciencedirect.com/science/article/pii/S0031320319300950) `"The impact of class imbalance in classification performance metrics based on the binary confusion matrix"` details the list of available metrics for classification problems in general and shows what metrics work for imbalanced datasets

*The authors of the paper concludes that:*

> Null bias is also shown by two metrics directly depending on *SNS and SPC*: **geometric mean (GM) and bookmaker informedness (BM or single-threshold AUC)**. These solve the one-dimensionality problem of the SNS and SPC metrics by considering either their arithmetic (BM) or geometric (GM) mean. Although these two metrics constitute good alternatives to be used with imbalanced datasets, they have the drawback of **focusing on only the classification successes (λPP and λNN)**, and fail to directly consider the classification errors (λPN and λNP).

> The second best (lowest biased) cluster of metrics is that which is comprised of accuracy (ACC), Matthews correlation coefficient (MCC), and markedness (MK). These all have a global (not partial) perspective, since classification results on both positive and negative classes are considered. From among these 3 metrics, ACC focuses only on the classification successes, which is a drawback and, additionally, has the highest bias (except when extreme balanced datasets are used). In this cluster, the lowest bias is shown by MCC with moderate values (lower than 0.2 in the normalized version) for almost every value of the imbalance coefficient.

> Finally, the metrics in the third and fourth clusters, precision (PRC), negative predictive value (NPV), and F1 score (F1), are highly biased and should be avoided for use in imbalanced datasets. Table 14 summarizes the behaviour of performance metrics with imbalanced datasets.

*metrics calculation*

- SNS: Sensitivity = tp/(tp+fn), that is, successfully predicted 1 label divdied by all obs with 1 as ground truth label
- SPC: Specificity = tn/(tn+fp), that is, successfully predicted 0 label divdied by all obs with 0 as ground truth label
- GM: Geometric Mean = square root of (SNS*SPC)
- BM: Bookmaker Informedness = SNS + SPC - 1


<h3> XGBoost setup</h3>

[On this page](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html), it shows how to configurate XGBosst to deal with imbalanced dataset.

For common cases such as ads clickthrough log, the dataset is extremely imbalanced. This can affect the training of XGBoost model, and there are two ways to improve it.

- If you care only about the overall performance metric (AUC) of your prediction
  - Balance the positive and negative weights via scale_pos_weight
  - Use AUC for evaluation
- If you care about predicting the right probability
  - In such a case, you cannot re-balance the dataset
  - Set parameter max_delta_step to a finite number (say 1) to help convergence
  
  
**scale_pos_weight [default=1]**

- Control the balance of positive and negative weights, useful for unbalanced classes. 
- A typical value to consider: sum(negative instances) / sum(positive instances). 
- Also, see [Higgs Kaggle competition demo for examples](https://github.com/dmlc/xgboost/tree/master/demo/kaggle-higgs)

<h3>Precision-Recall-Gain Curves </h3>

- sklearn page: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
- related paper: http://people.cs.bris.ac.uk/~flach//PRGcurves/


<h3>f1 score </h3>
<a href=https://sebastianraschka.com/faq/docs/computing-the-f1-score.html>Here in his article</a>, Sebastian wrote the following: 



> - Finally, based on further simulations, Forman and Scholz concluded that the computation of F1TP, FP, FN (compared to the alternative ways of computing the F1 score), yielded the “most unbiased” estimate of the generalization performance using *k-fold cross-validation.*
> In any case, the bottom line is that we should not only choose the appropriate performance metric and cross-validation technique for the task, but we also take a closer look at how the different performance metrics are computed in case we cite papers or rely on off-the-shelve machine learning libraries.

>- the paper: Apples-to-Apples in Cross-Validation Studies: Pitfalls in Classifier Performance Measurement George Forman, Martin Scholz
https://www.hpl.hp.com/techreports/2009/HPL-2009-359.pdf


<h3>re-sampling </h3>

- Under-sampling and Over-sampling are the two popular categories. [Here](https://github.com/scikit-learn-contrib/imbalanced-learn) is the python package that facilitates the re-sampling.

- [On this stackexchange page](https://stats.stackexchange.com/questions/222558/classification-evaluation-metrics-for-highly-imbalanced-data), *geekoverdose* noted that "I've seen both cases where doing and not doing up-/or downsampling resulted in the better final outcomes, so this is something you might need to try out (but don't manipulate your test set(s)!)."

> Yes, your assumptions about Kappa seem about right. Kappa as single, scalar metrics is mostly and advantage over other single, scalar metrics like accuracy, which will not reflect prediction performance of smaller classes (shadowed by performance of any much bigger class). Kappa solves this problem more elegantly, as you pointed out.

> **Using a metric like Kappa to measure your performance will not necessarily increase how your model fits to the data.** You could measure the performance of any model using a number of metrics, but how the model fits data is determined using other parameters (e.g. hyperparameters). So you might use e.g. Kappa for selecting a best suited model type and hyperparametrization amongst multiple choices for your very imbalanced problem - but just computing Kappa itself will not change how your model fits your imbalanced data.

> For different metrics: besides Kappa and precision/recall, also take a look at true positive and true negative rates TPR/TNR, and ROC curves and the area under curve AUC. Which of those are useful for your problem will mostly depend on the details of your goal. For example, the different information reflected in TPR/TNR and precision/recall: is your goal to have a high share of frauds actually being detected as such, and a high share of legitimate transactions being detected as such, and/or minimizing the share of false alarms (which you will naturally get "en mass" with such problems) in all alarms?

> For up-/downsampling: I think there is no canonical answer to "if those are required". They are more one way of adapting your problem. Technically: yes, you could use them, but use them with care, especially upsampling (you might end up creating unrealistic samples without noticing it) - and be aware that changing the frequency of samples of both classes to something not realistic "in the wild" might have negative effects on prediction performance as well. **At least the final, held-out test set should reflect the real-life frequency of samples again.** Bottom line: I've seen both cases where doing and not doing up-/or downsampling resulted in the better final outcomes, so this is something you might need to try out (but don't manipulate your test set(s)!).

- Also on this aforementiond stack page, *Johnson* mentioned a few other metrics: F1 score, Geometric mean, and Jaccard index.
> - F1 score, which is the harmonic mean of precision and recall.
> - G-measure, which is the geometric mean of precision and recall. Compared to F1, I've found it a bit better for imbalanced data.
> - Jaccard index, which you can think of as the TP/(TP+FP+FN). This is actually the metric that has worked for me the best.
> - **Note: For imbalanced datasets, it is best to have your metrics be macro-averaged.**


- My own experience on re-sampling imbalanced data did not help improving overall performance and ended up turning to other approaches to improve models.


<h3>Matthews correlation coefficient (MCC) </h3>

In the paper "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation" (<a href=https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7> url</a>), the authors claimed that:

> The Matthews correlation coefficient (MCC) is a more reliable statistical rate which produces a high score only if the prediction obtained good results in all of the four confusion matrix categories (true positives, false negatives, true negatives, and false positives), proportionally both to the size of positive elements and the size of negative elements in the dataset.

>... **MCC produces a more informative and truthful score in evaluating binary classifications than accuracy and F1 score**, by first explaining the mathematical properties, and then the asset of MCC in six synthetic use cases and in a real genomics scenario. We believe that the Matthews correlation coefficient should be preferred to accuracy and F1 score in evaluating binary classification tasks by all scientific communities.

<h3> Neyman-Pearson (NP) paradigm </h3>

- This <a href=https://arxiv.org/pdf/2002.04592.pdf> paper </a> introduced something I haven't seen in many literatures. 
- on this github <a href=https://github.com/ZhaoRichard/nproc> link </a>, the author stated the following:
> In many binary classification applications, such as disease diagnosis and spam detection, practitioners commonly face the need to **limit type I error** (i.e., the conditional probability of misclassifying a class 0 observation as class 1) so that it remains below a desired threshold. To address this need, the Neyman-Pearson (NP) classification paradigm is a natural choice; **it minimizes type II error(i.e., the conditional probability of misclassifying a class 1 observation as class 0) while enforcing an upper bound, alpha, on the type I error.** Although the NP paradigm has a century-long history in hypothesis testing, it has not been well recognized and implemented in classification schemes. Common practices that directly limit the empirical type I error to no more than alpha do not satisfy the type I error control objective because the resulting classifiers are still likely to have type I errors much larger than alpha. As a result, the NP paradigm has not been properly implemented for many classification scenarios in practice. In this work, we develop the first umbrella algorithm that implements the NP paradigm for all scoring-type classification methods, including popular methods such as logistic regression, support vector machines and random forests. Powered by this umbrella algorithm, we propose a novel graphical tool for NP classification methods: NP receiver operating characteristic (NP-ROC) bands, motivated by the popular receiver operating characteristic (ROC) curves. NP-ROC bands will help choose in a data adaptive way and compare different NP classifiers.



<h3> How to choose the metric - a decision map </h3>

The authoer <a href=https://aidevelopmenthub.com/tour-of-evaluation-metrics-for-imbalanced-classification/> here </a> drew a decision tree on how to choose a metric for a classification machine learning project with imbalanced data. 

I highly recommend this post as it presents a nice visual aid.
