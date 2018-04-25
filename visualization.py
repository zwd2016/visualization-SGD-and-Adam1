# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:33:54 2018

@author: Wendong Zheng
"""
import tensorflow as tf  
import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
  
lr = 0.1  
real_params = [1.2, 2.5]   # 真正的参数  
tf_X = tf.placeholder(tf.float32, [None, 1])  
tf_y = tf.placeholder(tf.float32, [None, 1])  
weight = tf.Variable(initial_value=[[5]], dtype=tf.float32)  
bia = tf.Variable(initial_value=[[4]], dtype=tf.float32)  
y = tf.matmul(tf_X, weight) + bia  
  
loss = tf.losses.mean_squared_error(tf_y, y)
 
train_op1 = tf.train.GradientDescentOptimizer(lr).minimize(loss)#SGD  
train_op2 = tf.train.AdamOptimizer(lr).minimize(loss)#adam
 
X_data = np.linspace(-1, 1, 200)[:, np.newaxis]  
noise = np.random.normal(0, 0.1, X_data.shape)  
y_data = X_data * real_params[0] + real_params[1] + noise  
  
sess = tf.Session()  
sess.run(tf.global_variables_initializer())  

#figure 2-1 
weights1 = []  
biases1 = []  
losses1 = []  
for step in range(400):  
    w1, b1, cost1, _ = sess.run([weight, bia, loss, train_op1],  
                             feed_dict={tf_X: X_data, tf_y: y_data})  
    weights1.append(w1)  
    biases1.append(b1)  
    losses1.append(cost1)  
result1 = sess.run(y, feed_dict={tf_X: X_data, tf_y: y_data})

#figure 2-2
weights2 = []  
biases2 = []  
losses2 = []  
for step in range(400):  
    w2, b2, cost2, _ = sess.run([weight, bia, loss, train_op2],  
                             feed_dict={tf_X: X_data, tf_y: y_data})  
    weights2.append(w2)  
    biases2.append(b2)  
    losses2.append(cost2)  
result2 = sess.run(y, feed_dict={tf_X: X_data, tf_y: y_data})  

fig = plt.figure(2)  
ax_3d = Axes3D(fig)   
w_3d, b_3d = np.meshgrid(np.linspace(-2, 7, 30), np.linspace(-2, 7, 30))  
loss_3d = np.array(  
    [np.mean(np.square((X_data * w_ + b_) - y_data))  
     for w_, b_ in zip(w_3d.ravel(), b_3d.ravel())]).reshape(w_3d.shape)  
ax_3d.plot_surface(w_3d, b_3d, loss_3d, cmap=plt.get_cmap('rainbow'))  

weights1 = np.array(weights1).ravel()  
biases1 = np.array(biases1).ravel()

#figure 2-2
weights2 = np.array(weights2).ravel()  
biases2 = np.array(biases2).ravel()
  
ax_3d.view_init(55, 40)#view rotate
ax_3d.scatter(weights2[0], biases2[0], losses2[0], s=30, color='b')#figure 2-2  
ax_3d.scatter(weights1[0], biases1[0], losses1[0], s=30, color='r')#figure 2-1 
ax_3d.set_xlabel('w')  
ax_3d.set_ylabel('b') 
ax_3d.plot(weights2, biases2, losses2, lw=3, c='black',label='Adam') 
ax_3d.plot(weights1, biases1, losses1, lw=1, c='blue',label='SGD') 
ax_3d.legend(loc='upper left') 
plt.show()  