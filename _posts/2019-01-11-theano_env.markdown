---
layout:     post
title:      "Windows下Theano环境部署"
subtitle:   "Anaconda + Theano + GPU 安装"
author:     "逸杯久"
header-img: "img/post-bg-2019_1.jpg"
catalog: true
tags:
    - 神经网络
    - 机器学习
    - machine learning
    - neural network
    - Anaconda
    - Theano
    - gpu
    - cuda
    - cudnn
---

> “Learn to walk before you run. ”


​    

[TOC]

# 1 Windows下 Anaconda + Theano + GPU 安装

**Theano 是机器学习常用的一个库，由于历史原因，github上相当一部分机器学习的教程和项目采用该库。就我个人而言，网上常见的在 Windows 下搭建 Theano 环境的方法都不够简洁，并且容易出现错误。所以我整理了一份我我自己的安装笔记。**

## 1.1 CPU版本环境搭建方法

1. 从 <https://www.anaconda.com/download> 安装最新版本的 Anaconda。

2. 安装完毕后，从“开始”菜单中打开 **Anaconda Prompt**，然后输入以下命令添加清华mirror，加快下载速度（有VPN的土豪忽略此步骤）：

```shell
# 添加conda的清华镜像，配置文件在C:\Users\用户名\.condarc
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

# 设置搜索时显示通道地址
conda config --set show_channel_urls yes
```

3. 输入以下命令创建环境，环境名：“mlcc_theano”：

```shell
# 创建名叫“mlcc_theano”的Python3.6环境
conda create -n mlcc_theano pip python=3.6

# 进入“mlcc_theano”的Python3.6环境，同理：把"activate"替换成“deactivate”为退出
conda activate mlcc_theano

# 安装 theano 和 pygpu
conda install theano pygpu

# 使用pip 安装一些常用的库
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --ignore-installed --upgrade matplotlib pandas sklearn scipy seaborn

# 更新所有安装包，避免安装包之间版本不匹配。比如我久碰到各种包报错。
conda update -c anaconda --all
```

4. 当所有软件包安装完毕后，从“开始”菜单中打开 **Anaconda Navigator**。在 **Navigator** 中：

   1. 切换到 `mlcc_theano 环境，如以下屏幕截图所示。每次打开 Jupyter 时，都必须选择 `mlcc_theano ` 环境。

   2. 在 `mlcc_theano 环境中安装 `notebook`

   3. 安装 `notebook` 后，点击 **Launch**。此时将打开一个网络浏览器。

   4. 接下来在创建一个新的Python文件，运行以下代码：

      ```
      from theano import function, config, shared, tensor
      import numpy
      import time
      
      vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
      iters = 1000
      
      rng = numpy.random.RandomState(22)
      x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
      f = function([], tensor.exp(x))
      print(f.maker.fgraph.toposort())
      t0 = time.time()
      for i in range(iters):
          r = f()
      t1 = time.time()
      print("Looping %d times took %f seconds" % (iters, t1 - t0))
      print("Result is %s" % (r,))
      if numpy.any([isinstance(x.op, tensor.Elemwise) and
                    ('Gpu' not in type(x.op).__name__)
                    for x in f.maker.fgraph.toposort()]):
          print('Used the cpu')
      else:
          print('Used the gpu')
      ```

      输出如下：

      ```Python
      [Elemwise{exp,no_inplace}(<TensorType(float64, vector)>)]
      Looping 1000 times took 11.866616 seconds
      Result is [1.23178032 1.61879341 1.52278065 ... 2.20771815 2.29967753 1.62323285]
      Used the cpu
      
      Process finished with exit code 0
      
      ```

      

      ## 1.2 GPU版本环境搭建方法

      *common sense: GPU比CPU更适合tensor的计算，如果你电脑上有一块不错的显卡，那么用GPU计算，会更快得到结果。网上其他的安装方案太麻烦了，需要自己匹配显卡驱动，找对应的*cuda和cudnn。

      **前提：先安装好CPU版本的环境**

      1. 从“开始”菜单中打开 **Anaconda Prompt**，然后输入以下命令：

         ```shell
         # 安装cuda
         conda install -c anaconda cudatoolkit
         
         # 安装cudnn
         conda install -c anaconda cudnn
         
         # 更新所有的库
         conda update -c anaconda --all
         ```

      2. 在 `C:\Users\用户名` 下创建文件 `.theanorc.txt`，文件内容如下：

         ```txt
         [global]
         device = cuda
         floatX = float32
         ```

         

      3. 安装完成后运行一下上面的测试程序，结果如下：

      ```Python
      Mapped name None to device cuda: GeForce GTX 1070 with Max-Q Design (0000:01:00.0)
      [GpuElemwise{exp,no_inplace}(<GpuArrayType<None>(float32, vector)>), HostFromGpu(gpuarray)(GpuElemwise{exp,no_inplace}.0)]
      Looping 1000 times took 0.343786 seconds
      Result is [1.2317803 1.6187935 1.5227807 ... 2.2077181 2.2996776 1.623233 ]
      Used the gpu
      ```




##  参考资料：

- [Theano手册](http://deeplearning.net/software/theano/install_windows.html#installation)


















~~~~~~~~~~

~~~~~~~~~~