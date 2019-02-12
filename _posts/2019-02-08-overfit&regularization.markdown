---
layout:     post
title:      "改进神经⽹络的学习⽅法（二）"
subtitle:   "过拟合 和 正则化"
author:     "逸杯久"
header-img: "img/post-bg-2019_1.jpg"
catalog: true
tags:
    - 神经网络
    - 过拟合
    - 正则化
    - overfit
    - regularization
    - neural network
---

> “Action speak louder than words. ”



[TOC]

# 1 过拟合（Overfitting）

## 1.1 什么是过拟合

​	Google机器学习给出的解释是：创建的模型与**训练数据**过于匹配，以致于模型无法根据新数据做出正确的预测。

**下面我们看一个线性模型的例子：**

如下一个简单的训练数据集，我们为其建立模型：

​	![overfit_01]({{ site.url }}/img/NN/overfit&regularization/overfit_01.png)

我们分别建立一个**线性模型**（下图蓝色实线所示）y = kx 和一个 6阶**多项式模型**（下图橙色虚线所示），得到如下图所示的结果：

![overfit_02]({{ site.url }}/img/NN/overfit&regularization/overfit_02.png)

多项式模型比线性模型在训练数据上表现要更好，但是我们现在把这两个模型用在**测试数据**（下图“十字”为测试样本）上，做最后的验收评估，得到如下图所示的结果：

![overfit_03]({{ site.url }}/img/NN/overfit&regularization/overfit_03.png)

综上：这个例子中虽然 6阶多项式模型在训练数据测试中表现比线性模型好，但是应用到测试数据（实际预测）中变现并不好——这是由于模型与**训练数据**过于匹配，反而局限了模型对新数据的处理。



## 1.2 如何判断过拟合

​	发生过拟合时，训练集和验证集相对于训练迭代次数的损失通常如下**泛化曲线**所示：

![è®­ç"éçæå¤±å½æ°éæ¸ä¸éãç¸æ¯ä¹ä¸ï¼éªè¯éçæå¤±å½æ°åä¸éï¼ç¶åå¼å§ä¸åã](https://developers.google.cn/machine-learning/crash-course/images/RegularizationTwoLossFunctions.svg)

​	上图显示的是某个模型的训练损失逐渐减少，但验证损失最终增加。

**防止过拟合的方法有：**

- 正则化
- 扩大训练集
- 早停法
- 弃权



# 2 正则化（regularization）

​	奥卡姆的威廉是 14 世纪一位崇尚简单的修士和哲学家。他认为科学家应该优先采用更简单（而非更复杂）的公式或理论。**奥卡姆剃刀定律**在机器学习方面的运用如下：
​	机器学习模型越简单，良好的实证结果就越有可能不仅仅基于样本的特性。

​	正则化表达式通常如下所示：

​	
$$
\begin{eqnarray}  C = C_0 +  \frac{\lambda}{n} \sum_w f(w)
\tag{1}\end{eqnarray}
$$
*注:$$C_0$$是原始的代价函数 ，后面部分是**规范化项 **，衡量模型的复杂度（常用的有L1，L2正则化）。其中 λ > 0 可以称为规范化参数，⽽ n 就是训练集合的⼤⼩； sgn(w) 就是 w 的正负号，即 w 是正数时为 +1，⽽ w 为负数时为 -1。  *

## 2.1 L1正则化

L1正则化公式如下所示：


$$
\begin{eqnarray}  C = C_0 + \frac{\lambda}{n} \sum_w |w|.
\tag{2}\end{eqnarray}
$$


这个⽅法是在未规范化的代价函数上加上⼀个权重绝对值的和。

对公式（2）进行求导，可得：


$$
\begin{eqnarray}  \frac{\partial C}{\partial
    w} &=& \frac{\partial C_0}{\partial w} + \frac{\lambda}{n} \, {\rm
    sgn}(w) 
  - \eta \frac{\partial C_0}{\partial w}
\tag{3}\end{eqnarray}
$$

$$
\begin{eqnarray}  \frac{\partial C}{\partial
    b} & = &\frac{\partial C_0}{\partial b} 
\tag{4}\end{eqnarray}
$$

根据公式（4）可知，参数b的更新不变

根据用公式（3）L1正则化的参数w更新公式如下：


$$
\begin{eqnarray}  w \rightarrow w' &=&
  w-\frac{\eta \lambda}{n} \mbox{sgn}(w) - \eta \frac{\partial
    C_0}{\partial w} \tag{5} \\
    &=& w （1- \frac{\eta \lambda}{n}）
  - \eta \frac{\partial C_0}{\partial w}
\tag{6}\end{eqnarray}
$$


## 2.2 L2正则化

L2正则化公式如下所示：


$$
\begin{eqnarray}  C = C_0 + \frac{\lambda}{2n}
\sum_w w^2,
\tag{8}\end{eqnarray}
$$


这个⽅法是在未规范化的代价函数上加上⼀个权重平方和。

对公式（8）进行求导，可得：



$$
\begin{eqnarray} 
  \frac{\partial C}{\partial w} & = & \frac{\partial C_0}{\partial w} + 
  \frac{\lambda}{n} w \tag{9}\\ 
  \frac{\partial C}{\partial b} & = & \frac{\partial C_0}{\partial b}
\tag{10}\end{eqnarray}
$$

根据公式（10）可知，参数b的更新不变

根据用公式（9）L2正则化的参数w更新公式如下：

$$
\begin{eqnarray} 
  w \rightarrow w' &=& w-\eta \frac{\partial C_0}{\partial
    w}-\frac{\eta \lambda}{n} w \tag{11}\\
  &=& \left(1-\frac{\eta \lambda}{n}\right) w -\eta \frac{\partial
    C_0}{\partial w}. 
\tag{12}\end{eqnarray}
$$

## 2.3 L1和L2的区别

​	在 L1 规范化中，权重通过⼀个常量向 0 进⾏缩⼩；在 L2 规范化中，权重通过⼀个和 w 成⽐例的量进⾏缩⼩的。

​	所以L1规范化会使部分w为0，达到对模型进行“降维”的效果；而L2只能部分w接近0，并不能达到“降维”的效果。



## 2.4 L2正则化代码示例

​	代码下载参考《[基于感知机的手写数字识别神经网络]({{site.url}}/2019/01/23/PNN/)》中的1.4部分。

​	运行下面代码：

```python
"""
overfitting
~~~~~~~~~~~

Plot graphs to illustrate the problem of overfitting.  
"""

# Standard library
import json
import random
import sys

# My library
sys.path.append('../src/')
import src.mnist_loader as mnist_loader
import src.network2 as network2

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np


def main(filename, num_epochs,
         training_cost_xmin=200, 
         test_accuracy_xmin=200, 
         test_cost_xmin=0, 
         training_accuracy_xmin=0,
         training_set_size=1000, 
         lmbda=0.0):
    """``filename`` is the name of the file where the results will be
    stored.  ``num_epochs`` is the number of epochs to train for.
    ``training_set_size`` is the number of images to train on.
    ``lmbda`` is the regularization parameter.  The other parameters
    set the epochs at which to start plotting on the x axis.
    """
    run_network(filename, num_epochs, training_set_size, lmbda)
    make_plots(filename, num_epochs, 
               training_cost_xmin,
               test_accuracy_xmin,
               test_cost_xmin, 
               training_accuracy_xmin,
               training_set_size)
                       
def run_network(filename, num_epochs, training_set_size=1000, lmbda=0.0):
    """Train the network for ``num_epochs`` on ``training_set_size``
    images, and store the results in ``filename``.  Those results can
    later be used by ``make_plots``.  Note that the results are stored
    to disk in large part because it's convenient not to have to
    ``run_network`` each time we want to make a plot (it's slow).

    """
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost())
    net.large_weight_initializer()
    test_cost, test_accuracy, training_cost, training_accuracy \
        = net.SGD(training_data[:training_set_size], num_epochs, 10, 0.5,
                  evaluation_data=test_data, lmbda = lmbda,
                  monitor_evaluation_cost=True, 
                  monitor_evaluation_accuracy=True, 
                  monitor_training_cost=True, 
                  monitor_training_accuracy=True)
    f = open(filename, "w")
    json.dump([test_cost, test_accuracy, training_cost, training_accuracy], f)
    f.close()

def make_plots(filename, num_epochs, 
               training_cost_xmin=200, 
               test_accuracy_xmin=200, 
               test_cost_xmin=0, 
               training_accuracy_xmin=0,
               training_set_size=1000):
    """Load the results from ``filename``, and generate the corresponding
    plots. """
    f = open(filename, "r")
    test_cost, test_accuracy, training_cost, training_accuracy \
        = json.load(f)
    f.close()
    plot_training_cost(training_cost, num_epochs, training_cost_xmin)
    plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin)
    plot_test_cost(test_cost, num_epochs, test_cost_xmin)
    plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size)
    plot_overlay(test_accuracy, training_accuracy, num_epochs,
                 min(test_accuracy_xmin, training_accuracy_xmin),
                 training_set_size)

def plot_training_cost(training_cost, num_epochs, training_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_epochs), 
            training_cost[training_cost_xmin:num_epochs],
            color='#2A6EA6')
    ax.set_xlim([training_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the training data')
    plt.show()

def plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_accuracy_xmin, num_epochs), 
            [accuracy/100.0 
             for accuracy in test_accuracy[test_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([test_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the test data')
    plt.show()

def plot_test_cost(test_cost, num_epochs, test_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_cost_xmin, num_epochs), 
            test_cost[test_cost_xmin:num_epochs],
            color='#2A6EA6')
    ax.set_xlim([test_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the test data')
    plt.show()

def plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_accuracy_xmin, num_epochs), 
            [accuracy*100.0/training_set_size 
             for accuracy in training_accuracy[training_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([training_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the training data')
    plt.show()

def plot_overlay(test_accuracy, training_accuracy, num_epochs, xmin,
                 training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(xmin, num_epochs), 
            [accuracy/100.0 for accuracy in test_accuracy], 
            color='#2A6EA6',
            label="Accuracy on the test data")
    ax.plot(np.arange(xmin, num_epochs), 
            [accuracy*100.0/training_set_size 
             for accuracy in training_accuracy], 
            color='#FFA933',
            label="Accuracy on the training data")
    ax.grid(True)
    ax.set_xlim([xmin, num_epochs])
    ax.set_xlabel('Epoch')
    ax.set_ylim([90, 100])
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    filename = input("Enter a file name: ")
    num_epochs = int(input(
        "Enter the number of epochs to run for: "))
    training_cost_xmin = int(input(
        "training_cost_xmin (suggest 200): "))
    test_accuracy_xmin = int(input(
        "test_accuracy_xmin (suggest 200): "))
    test_cost_xmin = int(input(
        "test_cost_xmin (suggest 0): "))
    training_accuracy_xmin = int(input(
        "training_accuracy_xmin (suggest 0): "))
    training_set_size = int(input(
        "Training set size (suggest 1000): "))
    lmbda = float(input(
        "Enter the regularization parameter, lambda (suggest: 5.0): "))
    main(filename, num_epochs, training_cost_xmin, 
         test_accuracy_xmin, test_cost_xmin, training_accuracy_xmin,
         training_set_size, lmbda)

```

输出如下：

```python
Enter a file name: overfitting
Enter the number of epochs to run for: 400
training_cost_xmin (suggest 200): 200
test_accuracy_xmin (suggest 200): 200
test_cost_xmin (suggest 0): 0
training_accuracy_xmin (suggest 0): 0
Training set size (suggest 1000): 1000
Enter the regularization parameter, lambda (suggest: 5.0): 0.1
```

可以的得到以下运行结果：

![myplot_2]({{ site.url }}/img/NN/overfit&regularization/myplot_2.png)

![myplot_1]({{ site.url }}/img/NN/overfit&regularization/myplot_1.png)

显然，规范化的使⽤能够解决过度拟合的问题。 



# 注意事项


​	**权重衰减因⼦** ：L2中$$\left(1-\frac{\eta \lambda}{n}\right) $$，在引入更大是训练数据集时，$$\lambda$$需要跟随n变大而变大，才能保证有参考意义。



# 其他防止过拟合的技术——弃权

​	**弃权（Dropout）**是⼀种相当激进的技术。和正则化不同，弃权技术并不依赖对代价函数的修改。⽽是，在弃权中，我们改变了⽹络本⾝。

​	弃权的想法是：三个臭皮匠胜过一个诸葛亮——一个神经网络容易产生过拟合，那么多个神经网络参与其中，进行“投票”选出结果，不就能避免过拟合了么？！然而其并没有建立多个神经网络，而是在同一个神经网络中隐藏层每一层随机丢弃一些权重，相当于这些节点丢失，从而构建不同结构的神经网络进行训练，这样最后我们将获得一个由多个“片面”的神经网络组合在一起，变得“全面”的神经网络——弃权充满争议，在此不做讨论。



## 参考资料：

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html)