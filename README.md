# CAR_STATUS-Analysis-1.0
Tensorflow

@[toc]
# 源码

完整代码已经上传到我的[Github](https://github.com/lixiang007666/CAR_STATUS-Analysis-1.0)！
# 1 分析数据




## 1.1 目标数据网站
[HERE！](http://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200502160908410.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgzODc4NQ==,size_16,color_FFFFFF,t_70)
数据位置：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200502160929586.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgzODc4NQ==,size_16,color_FFFFFF,t_70)

属性信息：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200502160810482.png)
## 1.2 数据信息（转成onehot）
previous：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200502161318186.png)
After:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200502161356794.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgzODc4NQ==,size_16,color_FFFFFF,t_70)

首先我们需要了解一下这一次所要用到的数据是什么. 这个车辆状况的数据是我在网上一个数据库找到的. 具体的数据描述也能在那个网页找到. 下面我来简单的说明一下.

车辆的状态分为四类:

unacc (Unacceptable 状况很差)
acc (Acceptable 状况一般)
good (Good 状况好)
vgood (Very good 状况非常好)
那我们又是通过什么来判断这辆车的状态好坏呢?

buying (购买价: vhigh, high, med, low)
maint (维护价: vhigh, high, med, low)
doors (几个门: 2, 3, 4, 5more)
persons (载人量: 2, 4, more)
lug_boot (贮存空间: small, med, big)
safety (安全性: low, med, high)
如果展示出这些数据, 我们就能清楚的看到这些数据的表示形式了.

buying	maint	doors	persons	lug_boot	safety	condition
vhigh	vhigh	2	2	small	low	unacc
vhigh	vhigh	2	2	small	med	unacc
vhigh	vhigh	2	2	small	high	unacc
好了, 第一个问题来了, 我们能不能直接这样喂给神经网络让它学习呢? 如果你有过一些处理神经网络的经历, 你就会察觉到, 这里的数据有很多都是文字形式的, 比如 vhigh, small 等. 而神经网络能够读取的数据形式都是数字. 这可怎么办? 或者说, 我们要通过什么样的途径, 将这些文字数据转换成数字呢?

接下来我介绍两种途径, 然后谈谈每一种的优缺点.

途径一

我们观察到这些文字描述不超过几个类别, 比如在 buying 下面, 总共也就这几种情况 (vhigh, high, med, low), 那我们能不能直接将每种情况给它一个数字代替呢? 比如 (vhigh=0, high=1, med=2, low=3).

途径二

同样是类别, 如果你听说过那个手写数字 MNIST 的数据集 (可以看看这个教程), 你会发现, 0/1/2/3/4/5/6/7/8/9 这十个数字不是直传入神经网络, 而是进行了一次 onehot 的处理. 也就是将数字变成只有 0/1 的形式, 比如下面:

0 -> [1,0,0,0,0,0,0,0,0,0]
1 -> [0,1,0,0,0,0,0,0,0,0]
2 -> [0,0,1,0,0,0,0,0,0,0]
…
9 -> [0,0,0,0,0,0,0,0,0,1]
转换的途径一般有这两种, 那我们选哪个? 途径一非常简单, 不同的类别变成不同的数字, 但是试想这样的情况, 如果现在是红色, 蓝色, 黄色这三个类别需要转换, 如果红色=0, 蓝色=1, 黄色=2, 红色到蓝色差了1, 红色到黄色差了2, 但是在真实的世界中, 各种颜色之间真的有数量差? 红色到蓝色的差别真的比红色到黄色大? 显然不是, 所以这样的类别转换数字的途径还是存在一定的问题.

而途径二, 我们如果转换成 onehot 形式, 红黄蓝它们就没有这种距离上的差距概念, 而是每个类别都是独立, 特别的, 不能互相比较的. 这才是比较好的类别转换形式. 我觉得有必要提的是, 像(vhigh=0, high=1, med=2, low=3)这样的类别, 可能还是存在一些从高到底的顺序, 这样的类别, 理论上也是可以使用途径一. 大家到了最后可以测试一下途径一和途径二的差别. 这个实战练习, 我是基于途径二的转换.

## 1.3 数据预处理

```python
def load_data(download=True):
    # download data from : http://archive.ics.uci.edu/ml/datasets/Car+Evaluation
    if download:
        data_path, _ = urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", "./car.csv")
        print("Downloaded to car.csv")

    # use pandas to view the data structure
    col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    data = pd.read_csv("car.csv", names=col_names)
    return data


def convert2onehot(data):
    # covert data to onehot representation
    return pd.get_dummies(data, prefix=data.columns)

```
**下载数据，并转换成onehot。**
这个功能, 让我们可以选择下载网上的数据 (大概51KB), 这个文件将会以 “car.csv” 文件名保存, 或者你已经下载好了, 将 download=False 直接载入本地数据.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200502161220464.png)
我们同样可以输出一下每类数据类型有多少, 检验一下是否和网上的描述一致.

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020050216123277.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgzODc4NQ==,size_16,color_FFFFFF,t_70)

```python
确认无误之后, 我们就开始使用上面提到的途径二来对这些类别数据做 onehot 预处理. 好在如果你使用 pandas, 它有一个很 handy 的功能 pd.get_dummies(), 来帮你实现 onehot 形式的数据转化.
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200502161253286.png)
# 2 搭建模型
## 2.1 导入数据

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import DATA_PRE

data = DATA_PRE.load_data(download=False)
new_data = DATA_PRE.convert2onehot(data)
# prepare training data
#pandas to numpy
new_data = new_data.values.astype(np.float32)       # change to numpy array and float32
np.random.shuffle(new_data)                         # Disorder
sep = int(0.7*len(new_data))
train_data = new_data[:sep]                         # training data (70%)
test_data = new_data[sep:]                          # test data (30%)

```
## 2.2 搭建网络
接着我们就搭建神经网络, input 数据的后面4个是真实数据的4类型的 onehot 形式. 我们添加两层隐藏层, 用 softmax 来输出每种类型的概率. 使用 tensorflow 的功能计算 loss 和 accuracy.

```python
# build network
tf_input = tf.placeholder(tf.float32, [None, 25], "input")
tfx = tf_input[:, :21]
tfy = tf_input[:, 21:]

l1 = tf.layers.dense(tfx, 128, tf.nn.relu, name="l1")
l2 = tf.layers.dense(l1, 128, tf.nn.relu, name="l2")
out = tf.layers.dense(l2, 4, name="l3")
prediction = tf.nn.softmax(out, name="pred")

loss = tf.losses.softmax_cross_entropy(onehot_labels=tfy, logits=out)
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(out, axis=1),)[1]
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = opt.minimize(loss)

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200502161808202.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgzODc4NQ==,size_16,color_FFFFFF,t_70)
## 2.3 训练网络
搭建好图, 然后通过 tensorboard 检查一下有没有错误, 最后就能开始训练啦. 通过4000次循环, 这里我们使用 Mini-batch update, 先随机生成 batch 的索引, 然后在 train_data 中选择数据当作这次的 batch. 这样运算起来比较快. 还有更快的方式, 比如使用 epoch 在每次 epoch 的时候 shuffle 一次数据, 然后在这个 shuffle 完的数据中按先后索引 batch 数据. 这都是使用 numpy 进行 mini-batch 运行速度上的经验之谈了.

```python
for t in range(4000):
    # training
    batch_index = np.random.randint(len(train_data), size=32)
    sess.run(train_op, {tf_input: train_data[batch_index]})

    if t % 50 == 0:
        # testing
        acc_, pred_, loss_ = sess.run([accuracy, prediction, loss], {tf_input: test_data})
        accuracies.append(acc_)
        steps.append(t)
        print("Step: %i" % t,"| Accurate: %.2f" % acc_,"| Loss: %.2f" % loss_,)
```

## 2.5 可视化学习过程

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200502162045406.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgzODc4NQ==,size_16,color_FFFFFF,t_70)
