# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:55:30 2018

@author: Bllue
"""
import os
import cv2
import numpy as np
import tensorflow as tf


img_width = 100
img_height = 100
train_percentage = 95
batch_size = 256

vgg16_npy_path = './model/vgg16.npy'
vgg_mean = [103.939, 116.779, 123.68]


train_datapath = []
train_label = []
test_datapath = []
test_label = []

filename = os.listdir('data/')
for i,path in enumerate(filename):
   dataname  = os.listdir('data/'+path)
   for file in dataname:
       chance = np.random.randint(100)
       if chance < train_percentage:
           train_datapath.append('data/'+path+'/'+file)
           train_label.append(i)
       else:
           test_datapath.append('data/'+path+'/'+file)
           test_label.append(i)
print('train data:',len(train_datapath))
print('test data:',len(test_datapath))

# 加载训练数据
train_data = np.zeros([len(train_datapath),img_width,img_height,3])
train_data = train_data.astype(np.uint8)
train_label_onehot = np.zeros([len(train_datapath),len(filename)])

i = 0
for path in train_datapath:
   img = cv2.imread(train_datapath[i])
   train_data[i,:,:,:]  = cv2.resize(img,(img_height,img_width))
   train_label_onehot[i,int(train_label[i])] = 1 
   i += 1


# 加载测试数据
test_data = np.zeros([len(test_datapath),img_width,img_height,3])
test_data = test_data.astype(np.uint8)
test_label_onehot = np.zeros([len(test_datapath),len(filename)])

i = 0
for path in test_datapath:
   img = cv2.imread(path)
   test_data[i,:,:,:]  = cv2.resize(img,(img_height,img_width))
   test_label_onehot[i,test_label[i]] = 1 
   i += 1
   
print('shape of train data:',train_data.shape)
print('shape of train label:',train_label_onehot.shape)
print('shape of test data:',test_data.shape)
print('shape of test label:',test_label_onehot.shape)

data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
#for key in data_dict.keys():
#    print(key)
#    print(len(data_dict[key]))
#    print(data_dict[key][0].shape, data_dict[key][1].shape)
def max_pool(bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def conv_layer(bottom, name):
    with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
        conv = tf.nn.conv2d(bottom, data_dict[name][0], [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, data_dict[name][1]))
        return lout
        
with tf.variable_scope('input'):        
    tfx = tf.placeholder(tf.float32, [None, 100, 100, 3])
    tfy = tf.placeholder(tf.float32, [None, 5])
    
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=tfx * 255.0)
    bgr = tf.concat(axis=3, values=[
        blue - vgg_mean[0],
        green - vgg_mean[1],
        red - vgg_mean[2],
    ])
    
conv1_1 = conv_layer(bgr, "conv1_1")
conv1_2 = conv_layer(conv1_1, "conv1_2")
pool1 = max_pool(conv1_2, 'pool1')

conv2_1 = conv_layer(pool1, "conv2_1")
conv2_2 = conv_layer(conv2_1, "conv2_2")
pool2 = max_pool(conv2_2, 'pool2')

conv3_1 = conv_layer(pool2, "conv3_1")
conv3_2 = conv_layer(conv3_1, "conv3_2")
conv3_3 = conv_layer(conv3_2, "conv3_3")
pool3 = max_pool(conv3_3, 'pool3')

conv4_1 = conv_layer(pool3, "conv4_1")
conv4_2 = conv_layer(conv4_1, "conv4_2")
conv4_3 = conv_layer(conv4_2, "conv4_3")
pool4 = max_pool(conv4_3, 'pool4')

conv5_1 = conv_layer(pool4, "conv5_1")
conv5_2 = conv_layer(conv5_1, "conv5_2")
conv5_3 = conv_layer(conv5_2, "conv5_3")
pool5 = max_pool(conv5_3, 'pool5')

# flatten = tf.reshape(pool5, [-1, 7*7*512])
flatten = tf.reshape(pool5, [-1, 4*4*512])

# fc6 = tf.layers.dense(flatten, 256, tf.nn.relu, name='fc6')
# out = tf.layers.dense(fc6, 5, tf.nn.softmax, name='out')

# out = tf.layers.dense(flatten,5,name='out')

with tf.name_scope('final_training_ops'):
    weights = tf.Variable(
        tf.truncated_normal(
            [4*4*512, 5], stddev=0.1))
    biases = tf.Variable(tf.zeros([5]))
    logits = tf.matmul(flatten, weights) + biases
    out = tf.nn.softmax(logits)

# loss = tf.losses.mean_squared_error(labels=tfy, predictions=out)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tfy, logits=logits)) 
# loss = tf.losses.softmax_cross_entropy(onehot_labels=tfy,logits=out)

# train_op = tf.train.RMSPropOptimizer(0.01).minimize(loss)
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
# train_op = tf.train.AdamOptimizer().minimize(loss)


correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(tfy, 1))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
            
            
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  
    writer = tf.summary.FileWriter('./logs', sess.graph)
    for it in range(10):
        # print(i)
        for i in range(int(len(train_datapath)/batch_size)):
            l,_ = sess.run([loss,train_op], {tfx:train_data[batch_size*i:batch_size*(i+1)], tfy: train_label_onehot[batch_size*i:batch_size*(i+1)]})
            print(i,'loss:',l)
        
        acc = sess.run(accuracy,feed_dict={tfx:test_data , tfy:test_label_onehot})
        print('it:',it,' acc:',acc)

# #        for i in range(int(len(train_datapath)/batch_size)):
# #        sess.run(train_op,feed_dict={tfx: train_data[batch_size*i : batch_size*(i+1)], tfy: train_label_onehot[batch_size*i:batch_size*(i+1)]})
