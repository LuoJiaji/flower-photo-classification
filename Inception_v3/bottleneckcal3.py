# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:50:00 2018

@author: Bllue
"""

import os
import random
import numpy as np
import tensorflow as tf


datapath = 'data/tmp/bottleneck/'

n_calss = len(os.listdir(datapath))
batchsize = 256

def get_datalist(datapath,train_percentage=80,test_percentage =10):
    train_datapath = []
    train_label = []
    test_datapath = []
    test_label = []
    validation_datapath = []
    validation_label = []

    filename = os.listdir(datapath)
    for i,path in enumerate(filename):
       dataname  = os.listdir(datapath+path)
       print(path,len(dataname))
       for file in dataname:
           chance = np.random.randint(100)
           if chance < train_percentage:
               train_datapath.append(datapath + path+ '/' + file)
               train_label.append(i)
           elif chance<(train_percentage+test_percentage):
               test_datapath.append(datapath + path+ '/' + file)
               test_label.append(i)
           else:
               validation_datapath.append(datapath + path + '/' + file)
               validation_label.append(i)
    print('train data:',len(train_datapath))
    print('test data:',len(test_datapath))
    return [train_datapath,train_label,test_datapath,test_label,validation_datapath,validation_label]




def get_random_batch(train_datapath,train_label,batchsize,n_class):
    # train_data = np.zeros([batchsize,2048])
    # train_data =  train_data.astype(np.uint8)
    # train_label_onehot = np.zeros([batchsize,n_calss])
    train_data = []
    train_label_onehot = []
    
    l = len(train_datapath)
    i = 0
    for _ in range(batchsize):
        # image_index = random.randrange(l)
        image_index = random.randrange(65535)
        image_index = image_index % len(train_datapath)  # 规范图片的索引

        with open(train_datapath[image_index], 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        # train_data[i,:] = bottleneck_values
        # train_label_onehot[i,int(train_label[image_index])] = 1
        train_data.append(bottleneck_values)
        label = np.zeros(n_class, dtype=np.float32)
        label[int(train_label[image_index])] = 1.0

        train_label_onehot.append(label )

#        print(i,image_index,train_datapath[image_index])
        i += 1
    return train_data,train_label_onehot

def get_test_data(test_datapath,test_label,n_class):
#    test_data = np.zeros([len(test_datapath),2048])
    # test_data = test_data.astype(np.uint8)
#    test_label_onehot = np.zeros([len(test_datapath),n_calss])
    
    test_data = []
    test_label_onehot = []
    
    i = 0
    for path in test_datapath:
        
        with open(path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
#        test_data[i,:]  = bottleneck_values
#        test_label_onehot[i,test_label[i]] = 1 
        test_data.append(bottleneck_values)
        label = np.zeros(n_class, dtype=np.float32)
        label[test_label[i]] = 1.0
        test_label_onehot.append(label)
         
        i += 1
    return test_data,test_label_onehot

#bottleneck_path = train_datapath[0]
#
#with open(bottleneck_path, 'r') as bottleneck_file:
#    bottleneck_string = bottleneck_file.read()
#    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    




BOTTLENECK_TENSOR_SIZE = 2048
n_classes = 5

bottleneck_input = tf.placeholder(
           tf.float32, [None, BOTTLENECK_TENSOR_SIZE],
           name='BottleneckInputPlaceholder')

# 定义新的标准答案输入
ground_truth_input = tf.placeholder(
   tf.float32, [None, n_classes], name='GroundTruthInput')

# 定义一层全连接层解决新的图片分类问题
with tf.name_scope('fc1'):
   weights1 = tf.Variable(
       tf.truncated_normal(
           [BOTTLENECK_TENSOR_SIZE, 128], stddev=0.1))
   biases1 = tf.Variable(tf.zeros([128]))
   fc1 = tf.nn.relu(tf.matmul(bottleneck_input, weights1) + biases1)
   
with tf.name_scope('fc2'):
    weights2 = tf.Variable(tf.truncated_normal([128,n_classes], stddev=0.1))
    biases2 = tf.Variable(tf.zeros([n_classes]))
    logits = tf.matmul(fc1,weights2) + biases2
    final_tensor = tf.nn.softmax(logits)
   


# 定义交叉熵损失函数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
   logits=logits, labels=ground_truth_input)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy_mean)


# 计算正确率
with tf.name_scope('evaluation'):
   correct_prediction = tf.equal(
       tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
   evaluation_step = tf.reduce_mean(
       tf.cast(correct_prediction, tf.float32))


train_datapath,train_label,test_datapath,test_label,validation_datapath,validation_label = get_datalist(datapath)

train_data,train_label_onehot = get_random_batch(train_datapath,train_label,256,n_calss)
test_data,test_label_onehot = get_test_data(test_datapath,test_label,n_calss)


STEPS = 6000
   
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())  
   train_data,train_label_onehot = get_random_batch(train_datapath,train_label,256,n_calss)

   for i in range(STEPS):
       train_data,train_label_onehot = get_random_batch(train_datapath,train_label,256,n_calss)
       sess.run( train_step,feed_dict={bottleneck_input: train_data, ground_truth_input: train_label_onehot })
   #        print(i)
       
       if i % 100 == 0 or i + 1 == STEPS:
           test_accuracy = sess.run(evaluation_step,feed_dict={bottleneck_input: test_data,ground_truth_input: test_label_onehot})
           print(i,test_accuracy)


