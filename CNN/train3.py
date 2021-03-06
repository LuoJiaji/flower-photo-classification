# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:52:19 2018
取消批大小固定限制，固定模型全连接节点个数
根据批大小随机读取训练样本
@author: Bllue
"""
import os 
import cv2
import random
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
batch_size = 256
fc = 3136
train_percentage = 95

datapath = 'data/'
n_calss = len(os.listdir(datapath))


def get_datalist(datapath,train_percentage=90):
    train_datapath = []
    train_label = []
    test_datapath = []
    test_label = []

    filename = os.listdir(datapath)
    for i,path in enumerate(filename):
       dataname  = os.listdir(datapath+path)
       print(path,len(dataname))
       for file in dataname:
           chance = np.random.randint(100)
           if chance < train_percentage:
               train_datapath.append(datapath + path+ '/' + file)
               train_label.append(i)
           else:
               test_datapath.append(datapath + path + '/' + file)
               test_label.append(i)
    print('train data:',len(train_datapath))
    print('test data:',len(test_datapath))
    return [train_datapath,train_label,test_datapath,test_label]


def get_random_batch(train_datapath,train_label,batchsize,n_calss,img_width,img_height):
    
    train_data = np.zeros([batchsize,img_width,img_height,3])
    train_data =  train_data.astype(np.uint8)
    train_label_onehot = np.zeros([batchsize,n_calss])
    
    l = len(train_datapath)
    i = 0
    for _ in range(batchsize):
        image_index = random.randrange(l)
        img = cv2.imread(train_datapath[image_index])
        train_data[i,:,:,:]  = cv2.resize(img,(img_height,img_width))
        train_label_onehot[i,int(train_label[image_index])] = 1
#        print(i,image_index,train_datapath[image_index])
        i += 1
    return train_data,train_label_onehot


def get_test_data(test_datapath,test_label,n_calss,img_width,img_height):
    test_data = np.zeros([len(test_datapath),img_width,img_height,3])
    test_data = test_data.astype(np.uint8)
    test_label_onehot = np.zeros([len(test_datapath),n_calss])
    i = 0
    for path in test_datapath:
       img = cv2.imread(path)
       test_data[i,:,:,:]  = cv2.resize(img,(img_height,img_width))
       test_label_onehot[i,test_label[i]] = 1 
       i += 1
    return test_data,test_label_onehot

# 加载文件路径
train_datapath,train_label,test_datapath,test_label = get_datalist(datapath,train_percentage)

# 加载测试数据
test_data,test_label_onehot = get_test_data(test_datapath,test_label,n_calss,img_width,img_height)


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 100,100,3] )  
    y_ = tf.placeholder(tf.float32, shape=[None, 5])

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
  

  
# 全连接层  
with tf.name_scope('fc1'):
    # reshape = tf.reshape(L2_pool, shape=[batch_size, -1])
    # dim = reshape.get_shape()[1].value
    # print(dim)
    # print(reshape)

    W_fc1 = tf.Variable(tf.truncated_normal([fc, 1024], stddev=0.1))  
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))  
    
    h_pool2_flat = tf.reshape(L2_pool, [-1, fc])  
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
    # y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) 
    out = tf.nn.softmax(y_conv)

    tf.summary.histogram('W_fc2',W_fc2)  
  
# 定义优化器和训练op  
with tf.name_scope('loss_train'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))  
    # train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy) 
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy) 
    # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)


    tf.summary.scalar('loss',cross_entropy)

with tf.name_scope('acc'):     
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1))  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
    # tf.summary.scalar('acc',accuracy)

  
index = np.arange(0,len(train_datapath))
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

    for it in range(10):
        for i in range(int(len(train_datapath)/batch_size)):
            train_data,train_label_onehot = get_random_batch(train_datapath,train_label,256,n_calss,img_width,img_height)

            _, loss_ =  sess.run([train_step,cross_entropy],feed_dict={x: train_data, y_: train_label_onehot, keep_prob: 0.5})
#             train_step.run(feed_dict={x: data[0+512*i : 512+512*i], y_: label_onehot[0+512*i:512+512*i], keep_prob: 0.5})
#         iterate_accuracy = accuracy.eval(feed_dict={x: data[0:512], y_: label_onehot[0:512], keep_prob: 1.0})  
#             print(i,loss_)
            # summary = sess.run(merged,feed_dict={x: train_data, y_: train_label_onehot, keep_prob: 0.5})
            # writer.add_summary(summary,it*10+i)
             
        acc = sess.run(accuracy,feed_dict={x:train_data , y_:train_label_onehot, keep_prob: 1})
        # print(test_data.shape)
        # print(test_label_onehot.shape)
        acc_test = sess.run(accuracy,feed_dict={x:test_data , y_:test_label_onehot, keep_prob: 1})

        print('iter:', it ,'  train acc:', acc ,'  test acc:', acc_test)

        ckpt_path = './ckpt/model'
        # saver_path = saver.save(sess, ckpt_path, global_step=it)
        
       