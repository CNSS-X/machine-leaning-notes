#!/usr/bin/env python
#coding:utf-8
"""
  Author:  v1ll4n --<>
  Purpose: research the relationship between ad and money.
  Created: 11/07/17
"""

import tensorflow as tf
import numpy as np


# prepare your data set
money = np.array([[109],[82],[99], [72], [87], [78], [86], [84], [94], [57]]).astype(np.float32)
click = np.array([ [11], [8], [8],  [6],  [7],  [7],  [7],  [8],  [9],  [5]]).astype(np.float32)

# well, divide our data set into testing data and training data
X_test = money[0:5].reshape(-1,1)
Y_test = click[0:5]
X_train = money[5:].reshape(-1,1)
y_train = click[5:]



#
# build our model using tf
#
# tf setting placeholder
placeholder_x = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable(initial_value=tf.zeros([1,1]))
W = w
b = tf.Variable(initial_value=tf.zeros([1]))

y = tf.matmul(placeholder_x, w) + b

placeholder_y = tf.placeholder(tf.float32, [None, 1])

cost = tf.reduce_sum(tf.pow(placeholder_y - y, 2)) # (yi - f(xi))^2
train_step = tf.train.GradientDescentOptimizer(0.000001).minimize(cost)

#
# startup model
#
init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)
sess = session
cost_history = []

for i in range(1000):
    feed = {
        placeholder_x: X_train,
        placeholder_y: y_train
    }
    
    session.run(train_step, feed)
    cost_history.append(session.run(cost, feed_dict=feed))
    
    print('After {} iteration'.format(i))
    print('w: {} b: {} cost: {}'.format(session.run(w), session.run(b), session.run(cost, feed_dict=feed)))

print('w_value: {} b_value: {} cost: {}'.format(session.run(w), session.run(b), session.run(cost, feed)))