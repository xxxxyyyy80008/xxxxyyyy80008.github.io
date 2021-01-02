---
sort: 1
title: Temporal Convolutional Network (TCN)
---


### Notes on **Temporal Convolutional Network (TCN)**

***
1. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
[Accessed on Jan 2, 2021](https://arxiv.org/pdf/1803.01271.pdf)


- citation:
 > Bai, Shaojie & Kolter, J. & Koltun, Vladlen. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. 

- key takeaways:
  1. Our results indicate that a simple convolutiona architecture outperforms canonical recurrent networks such as LSTMs across a diverse range of tasks and datasets, while demonstrating longer effective memory. We conclude that the common association between sequence modeling and recurrent networks should be reconsidered, and convolutional networks should be regarded as a natural starting point for sequence modeling tasks. 
  1. Source code available at github: [accesses on Jan 2, 2021](https://github.com/locuslab/TCN)
  1. TCN outperforms LSTMs and vanilla RNNs by a significant margin in perplexity on LAMBADA, with a substantially smaller network and virtually no tuning.
  1. Due to the comparable clarity and simplicity of TCNs, we conclude that convolutional networks should be regarded as a natural starting point and a powerful toolkit for sequence modeling
  
