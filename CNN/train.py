# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:52:19 2018

@author: Bllue
"""
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


#filenames = os.walk('./')
img_width = 100
img_height = 100

#path =  os.walk('data/')
#for d in path:
#    print(d[0])
#    print(len(d[2]))


filename = os.listdir('data/')

datapath = []
label = []

for i,path in enumerate(filename):
    dataname  = os.listdir('data/'+path)
    for file in dataname:
        datapath.append('data/'+path+'/'+file)
        label.append(i)
        
temp = np.array([datapath, label])
temp = temp.transpose()     # 转置
np.random.shuffle(temp)        
image_list = temp[:, 0]
label_list = temp[:, 1]
    
#img = cv2.imread('./data/'+datapath[2872])
#cv2.imshow('a',img)

#img = cv2.resize(img,(img_width,img_height))
#cv2.imshow('src',img)
#cv2.waitKey()

data = np.zeros([len(datapath),img_width,img_height,3])
label_onehot = np.zeros([len(datapath),len(filename)])
data = data.astype(np.uint8)

i = 0 
for path in datapath:
    img = cv2.imread(datapath[i])
    data[i,:,:,:]  = cv2.resize(img,(img_height,img_width))
    label_onehot[i,label[i]] = 1 
    i +=1
    
    
#img = data[1000,:,:,:]
#img = img.astype(np.uint8)
#cv2.imshow('final',img )
#cv2.waitKey()
#plt.imshow(img)
#plt.show()


batch_size = 256
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[batch_size, 100,100,3])  
    y_ = tf.placeholder(tf.float32, shape=[batch_size, 5])

with tf.name_scope('conv1'):
    W_conv1 = tf.Variable(tf.truncated_normal([7, 7, 3, 32], stddev=0.1))  
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))  
    
    L1_conv = tf.nn.conv2d(x, W_conv1, strides=[1, 2, 2, 1], padding='SAME')  
    L1_relu = tf.nn.relu(L1_conv + b_conv1)  
    L1_pool = tf.nn.max_pool(L1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    tf.summary.histogram('W_conv1',W_conv1)
  
# 定义第二个卷积层的variables和ops  
with tf.name_scope('conv2'):
    W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))  
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))  
    
    L2_conv = tf.nn.conv2d(L1_pool, W_conv2, strides=[1, 2, 2, 1], padding='SAME')  
    L2_relu = tf.nn.relu(L2_conv + b_conv2)  
    L2_pool = tf.nn.max_pool(L2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    tf.summary.histogram('W_conv2',W_conv2)
  

fc = 3136
  
# 全连接层  
with tf.name_scope('fc1'):
    reshape = tf.reshape(L2_pool, shape=[batch_size, -1])
    dim = reshape.get_shape()[1].value
    print(dim)
    print(reshape)

    W_fc1 = tf.Variable(tf.truncated_normal([dim, 1024], stddev=0.1))  
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))  
    
    h_pool2_flat = tf.reshape(L2_pool, [-1, dim])  
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  
    tf.summary.histogram('W_fc1',W_fc1)
  
  
# dropout
with tf.name_scope('dropout'):  
    keep_prob = tf.placeholder(tf.float32)  
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  
  
  
# readout层
with tf.name_scope('out'):  
    W_fc2 = tf.Variable(tf.truncated_normal([1024, 5], stddev=0.1))  
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[5]))  
    
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    tf.summary.histogram('W_fc2',W_fc2)  
  
# 定义优化器和训练op  
with tf.name_scope('loss_train'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))  
    train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy) 
    tf.summary.scalar('loss',cross_entropy)

with tf.name_scope('acc'):     
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
    # tf.summary.scalar('acc',accuracy)

print('data:',data.shape)
print('label:',label_onehot.shape)
#  
index = np.arange(0,3000)
np.random.shuffle(index)

saver = tf.train.Saver()

with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
    writer = tf.summary.FileWriter('logs/',sess.graph)
    merged = tf.summary.merge_all()

    checkpoint = tf.train.get_checkpoint_state("ckpt")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    for it in range(200):
        for i in range(int(len(datapath)/batch_size)):

            _,loss_ =  sess.run([train_step,cross_entropy],feed_dict={x: data[batch_size*i : batch_size*(i+1)], y_: label_onehot[batch_size*i:batch_size*(i+1)], keep_prob: 0.5})
#             train_step.run(feed_dict={x: data[0+512*i : 512+512*i], y_: label_onehot[0+512*i:512+512*i], keep_prob: 0.5})
#         iterate_accuracy = accuracy.eval(feed_dict={x: data[0:512], y_: label_onehot[0:512], keep_prob: 1.0})  
#             print(i,loss_)
            summary = sess.run(merged,feed_dict={x: data[batch_size*i : batch_size*(i+1)], y_: label_onehot[batch_size*i:batch_size*(i+1)], keep_prob: 0.5})
            writer.add_summary(summary,it*10+i)
             
        acc = sess.run(accuracy,feed_dict={x:data[index[0:batch_size]] , y_:label_onehot[index[0:batch_size]], keep_prob: 1})
        acc_val = sess.run(accuracy,feed_dict={x:data[-batch_size:] , y_:label_onehot[-batch_size:], keep_prob: 1})

        print('iter:',it,"  acc:",acc,'  acc_val:',acc_val)

        ckpt_path = './ckpt/model'
        saver_path = saver.save(sess, ckpt_path, global_step=it)
        
#        