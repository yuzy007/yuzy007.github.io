---
layout:     post
title:      "深度神经网络梯度问题"
subtitle:   "梯度爆炸和梯度消失"
author:     "逸杯久"
header-img: "img/post-bg-2019_1.jpg"
catalog: true
tags:
    - 神经网络
    - 梯度爆炸
    - 梯度消失
    - vanishing gradient
    - exploding gradient
    - neural network
---

>Take control of your own destiny



[TOC]

、



# 1 梯度消失和梯度爆炸

​	早深层的神经网络中经常会遇到梯度消失和梯度爆炸问题，那么这是如何产生的呢？

## 1.1 举栗说明

​	这是个每一层只有一个神经元，深度为5的神经网络：

![example_01]({{ site.url }}/img/NN/vanishing&exploding_gradient/example_01.png)

​	这个网络采用 S 型激活函数，$$z_j = w_{j} a_{j-1}+b_j$$表示第j层的加权和，$$\sigma(z_j)$$表示第j层的输出，那么代价函数 C 对$$b_1$$的偏导表达式如下：

![example_02](http://neuralnetworksanddeeplearning.com/images/tikz38.png)

​	我们再看一下 S 型激活函数的密度函数$$\sigma'$$：

![example_03]({{ site.url }}/img/NN/vanishing&exploding_gradient/example_03.png)

​	可知：$\sigma'(z_j) \leq 1/4$；

如果用标准正态分布初始化$w_j$，那么大部分满足$\|w_j\| < 1$，从而导致：$\|w_j \sigma'(z_j)\| < 1/4$。

​	这样的神经网络**随着深度的增加，反向传递的结果会指数级的下降，从而导致梯度消失。**

​	那么梯度爆炸又是如何产生的呢？观察 $\|w_j \sigma'(z_j)\|$，会发现如果w足够大，比如设 w = 100 ,那么满足$\sigma'(z_j) > 1$，这样的神经网络随着深度的增加，**反向传递的结果会指数级的上升，从而导致梯度爆炸**。

## 1.2 解决办法

​	根据上面的例子，我们知道梯度问题的罪魁祸首是$\|w_j \sigma'(z_j)\|$，那么我们可以：

​	（a）使用ReLU 或者其他激励函数替换S 型激励函数。ReLU函数图如下：

![example_40](https://gss0.bdstatic.com/94o3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike92%2C5%2C5%2C92%2C30/sign=743e6d3efc1f3a294ec5dd9cf84cd754/0b55b319ebc4b745bbc37bf6c3fc1e178b8215ff.jpg)

​	可知ReLU大于0部分倒数恒为1，不会产生梯度丢失或者爆炸。同时该函数相比S型函数加简单，加快了正向传播和反向传播的速度，加速了网络的训练。

​	（b）初始化参数w时，注意满足$$\|w_j \sigma'(z_j) \| = 1$$，参考《[改进神经⽹络的学习⽅法（四）--超参数调节]({{ site.url }}/2019/02/12/hyperparameter/)》。

## 参考资料：

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html)
- [Google machine learning](https://developers.google.cn/machine-learning/crash-course/regularization-for-simplicity/playground-exercise-examining-l2-regularization)