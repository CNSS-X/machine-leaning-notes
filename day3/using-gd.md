# Example for GD

在了解了 GD 的理论之后，我们可以看一下具体这个东西是怎么用在昨天的例子中的。

```python
#!/usr/bin/env python
#coding:utf-8
"""
  Author:  v1ll4n --<>
  Purpose: research the relationship between ad and money.
  Created: 11/07/17
"""

import tensorflow
import numpy


# prepare your data set
money = numpy.array([[109],[82],[99], [72], [87], [78], [86], [84], [94], [57]]).astype(numpy.float32)
click = numpy.array([[11], [8], [8], [6],[ 7], [7], [7], [8], [9], [5]]).astype(numpy.float32)

# well, divide our data set into testing data and training data
X_test = money[0:5].reshape(-1,1)
Y_test = click[0:5]
X_train = money[5:].reshape(-1,1)
y_train = click[5:]

#
# build our model using tensorflow
# 这里是非常重要的部分吧，来我给大家翻译一下，PlaceHolder 就相当于是新建了一个未知量
# 未知量就是一个占位符，这个占位符是我们之后要传入数据的，就好像是我们写了一个方程，然后
# 把具体的数带入方程中进行计算。
#
# Variable 就是常量，建立了一个值，在下面的使用还带了初始值。
#
# tensorflow setting placeholder
placeholder_x = tensorflow.placeholder(tensorflow.float32, [None, 1])
w = tensorflow.Variable(initial_value=tensorflow.zeros([1,1]))
W = w # alias it
b = tensorflow.Variable(initial_value=tensorflow.zeros([1]))

# 这一步 matmul 一看就是 matrix multiple 啥的。。。矩阵相乘的意思
# 所以这里的意思也就是 y = f(x) = w * x + b
# 建立了一个函数。这个函数 w b 是我们想要求出来的具体的值。
# x（placeholder） 是我们要喂进去的数据
y = tensorflow.matmul(placeholder_x, w) + b

# 这里和 x 一样吧，准备建立一个占位符，等着喂 y 进来。
placeholder_y = tensorflow.placeholder(tensorflow.float32, [None, 1])

# 这里昨天我们只能理解到欧式距离之和，现在我们可以把它理解为一个 COST FUNCTION，
# 这个成本函数叫 cost 
# 当然严格对于 J 函数的定义来说，我们这里少了一个常数参数，但是无所谓，求极值是完全不影响的
cost = tensorflow.reduce_sum(tensorflow.pow(placeholder_y - y, 2)) # (yi - f(xi))^2

# 从这里开始准备进行定义步长
# 从函数的使用来说，是使用了 tensorflow 里训练算法的 GD 优化，使用步长为 0.000001 求成本函数最小
train_step = tensorflow.train.GradientDescentOptimizer(0.000001).minimize(cost)

#
# startup model 常规操作
#
init = tensorflow.global_variables_initializer()
session = tensorflow.Session()
session.run(init)
sess = session
cost_history = []

for i in range(1000):
    feed = {
        placeholder_x: X_train,
        placeholder_y: y_train
    }
    
    # 梯度下降一个步长
    session.run(train_step, feed)
    
    # 记录 and Debug IO
    cost_history.append(session.run(cost, feed_dict=feed))
    print('After {} iteration'.format(i))
    print('w: {} b: {} cost: {}'.format(session.run(w), session.run(b), session.run(cost, feed_dict=feed)))

print('w_value: {} b_value: {} cost: {}'.format(session.run(w), session.run(b), session.run(cost, feed)))
```

## LDA 

http://blog.csdn.net/aliceyangxi1987/article/details/75007855

这里是翻译的讲 LDA 不错的一篇文章，具体的计算步骤我们不打算进行计算了。