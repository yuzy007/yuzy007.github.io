---
layout:     post
title:      "改进神经⽹络的学习⽅法（四）"
subtitle:   "超参数调节"
author:     "逸杯久"
header-img: "img/post-bg-2019_1.jpg"
catalog: true
tags:
    - 神经网络
    - 加快训练
    - 超参数
    - hyperparameter
    - neural network
---

> “Don't aim for success if you want it; just do what you love and believe in, and it will come naturally.”



[TOC]

、



# 1 超参数（hyperparameter）



## 1.1 什么是超参数
> In [Bayesian statistics](https://en.wikipedia.org/wiki/Bayesian_statistics), a **hyperparameter** is a parameter of a [prior distribution](https://en.wikipedia.org/wiki/Prior_distribution); the term is used to distinguish them from parameters of the model for the underlying system under analysis.
>
> For example, if one is using a [beta distribution](https://en.wikipedia.org/wiki/Beta_distribution) to model the distribution of the parameter *p* of a [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution), then:
>
> - *p* is a parameter of the underlying system (Bernoulli distribution), and
> - *α* and *β* are parameters of the prior distribution (beta distribution), hence *hyper*parameters.
>
> One may take a single value for a given hyperparameter, or one can iterate and take a probability distribution on the hyperparameter itself, called a [hyperprior](https://en.wikipedia.org/wiki/Hyperprior).

——————以上内容来之wiki——————

​	Ok，上面一堆balabala的似乎有点抽象？那么你可以片面的理解为：在机器学习中，超参数是你设置模型时手动给出的参数（比如：$$\eta$$，$$\lambda$$），并非训练模型时候自动生成的参数（比如：w，b）。

​	超参数好比训练模型的“旋钮”，你可以通过调节超参数，更快地获得结果，甚至获得更好的结果。



## 1.2 超参数调节

​	*注：超参数调节目前没有统一的方法，网络上的资料大都是经验总结。*

### 1.2.1 宽泛策略

​	**宽泛策略**： 在使⽤神经⽹络来解决新的问题时，⼀个挑战就是获得任何⼀种⾮寻常的学习，也就是说，达到⽐随机的情况更好的结果。 

​	似乎很简单？但是如果训练数据很大，模型极其复杂，训练模型将耗时几天甚至几周。那么如何快速得到反馈呢？

​	我们可以这样做：

- 减少训练数据，比如：只用20%的训练数据进行验证，这样我们可以提高5倍的速度来得到结果
- 减少神经网络深度，加快训练结果
- 提高监控频率，比如：由原来每5000 次训练后反馈结果，提高成每1000 次。

**注意事项**：正则化参数λ  和训练数据大小正相关，调整模型时候，需要和训练数据一起做改变。

### 1.2.2 训练误差调整总结

**训练误差调整总结：**

- 训练误差应该稳步减小，刚开始是急剧减小，最终应随着训练收敛达到平稳状态。
- 如果训练尚未收敛，尝试运行更长的时间。
- 如果训练误差减小速度过慢，则提高学习速率也许有助于加快其减小速度。

- - 但有时如果学习速率过高，训练误差的减小速度反而会变慢。

- 如果训练误差变化很大，尝试降低学习速率。

- - 较低的学习速率和较大的步数/较大的批量大小通常是不错的组合。

- 批量大小过小也会导致不稳定情况。不妨先尝试 100 或 1000 等较大的值，然后逐渐减小值的大小，直到出现性能降低的情况。

重申一下，切勿严格遵循这些经验法则，因为效果取决于数据。请始终进行试验和验证。

### 1.2.3 超参数调整方法推荐

​	推荐方法：每次只调整一个超参数，确定好其中一个后再调整其他超参数。

——————来着《[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html)》——————

​	**规范化参数**： 我建议，开始时不包含规范化（λ = 0.0），确定 η 的值。使⽤确定出来的 η，我们可以使⽤验证数据来选择好的 λ。从尝试 λ = 1.0 开始，然后根据验证集上的性能10倍的增加或者减少。⼀旦我已经找到⼀个好的量级，你可以改进 λ 的值。这⾥搞定后，你就可以返回再重新优化 η。 

​	**学习速率调整**： 我们⼀直都将学习速率设置为常量。但是，通常采⽤可变的学习速率更加有效。在学习的前期，权重可能⾮常糟糕。所以最好是使⽤⼀个较⼤的学习速率让权重变化得更快。越往后，我们可以降低学习速率，这样可以作出更加精良的调整。 

​	**使⽤提前停⽌来确定训练的迭代期数量**： 正如我们在本章前⾯讨论的那样，提前停⽌表⽰在每个回合的最后，我们都要计算验证集上的分类准确率。 

​	**⼩批量数据⼤⼩**： 选择最好的⼩批量数据⼤⼩也是⼀种折衷。太⼩了，你不会⽤上很好的矩阵库的快速
计算。太⼤，你是不能够⾜够频繁地更新权重的。 



## 1.3 超参数调节在线实例

——————以下超参数调节实例来来自《[Google machine learning](https://developers.google.cn/machine-learning/crash-course/regularization-for-simplicity/playground-exercise-examining-l2-regularization)》——————

<div class="mlcc-scrollable-iframe-container">
  <iframe scrolling="no" style="width: 970px; height: 700px" class="inherit-locale" frameborder="0" src="https://developers.google.cn/machine-learning/crash-course/playground/?utm_source=engedu&amp;utm_medium=ss&amp;utm_campaign=mlcc&amp;hl=zh-cn#activation=linear&amp;regularization=L2&amp;batchSize=10&amp;dataset=gauss&amp;regDataset=reg-plane&amp;learningRate=0.03&amp;regularizationRate=0&amp;noise=50&amp;networkShape=&amp;seed=0.48288&amp;showTestData=false&amp;discretize=false&amp;percTrainData=10&amp;x=true&amp;y=true&amp;xTimesY=true&amp;xSquared=true&amp;ySquared=true&amp;cosX=false&amp;sinX=true&amp;cosY=false&amp;sinY=true&amp;collectStats=true&amp;problem=classification&amp;initZero=false&amp;hideText=true&amp;dataset_hide=false&amp;percTrainData_hide=false&amp;noise_hide=false&amp;batchSize_hide=false&amp;xTimesY_hide=false&amp;xSquared_hide=false&amp;ySquared_hide=false&amp;sinX_hide=false&amp;sinY_hide=false&amp;activation_hide=true&amp;learningRate_hide=false&amp;regularization_hide=false&amp;regularizationRate_hide=false&amp;numHiddenLayers_hide=true&amp;problem_hide=true&amp;tutorial=dp-regularization-for-simplicity-l2&amp;goalTrainTestDiffMaxThresholdFirst=0.085&amp;goalTrainTestDiffMinThresholdFirst=0.012"></iframe>
</div>
**答案部分**：

<p>将正则化率从 0 增至 0.3 会产生以下影响：</p>

<ul>
  <li><p>测试损失明显减少。</p>
  <p class="note">注意：虽然测试损失明显减少，训练损失实际上却有所增加。<em></em>这属于正常现象，因为您向损失函数添加了另一项来降低复杂度。最终，最重要的是测试损失，因为它是真正用于衡量模型能否针对新数据做出良好预测的标准。</p>
  </li>
  <li>测试损失与训练损失之间的差值明显减少。</li>
  <li>特征和某些特征组合的权重的绝对值较低，这表示模型复杂度有所降低。</li>
</ul>

<p>由于数据集具有随机性，因此无法预测哪个正则化率能得出最准确的结果。
对我们来说，正则化率为 0.3 或 1 时，一般测试损失降至最低。</p>

## 参考资料：

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html)
- [Google machine learning](https://developers.google.cn/machine-learning/crash-course/regularization-for-simplicity/playground-exercise-examining-l2-regularization)