# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 07:51:29 2018

@author: Bllue
"""
# import os
# import cv2
# import random
# import numpy as np
# import tensorflow as tf


# vgg16_npy_path = './model/vgg16.npy'
# train_percentage = 95
# img_width = 100
# img_height = 100


# #filename = os.listdir('data/')
# datapath = 'data/'
# n_calss = len(os.listdir(datapath))


# def get_datalist(datapath,train_percentage=90):
#     train_datapath = []
#     train_label = []
#     test_datapath = []
#     test_label = []

#     filename = os.listdir(datapath)
#     for i,path in enumerate(filename):
#        dataname  = os.listdir(datapath+path)
#        print(path,len(dataname))
#        for file in dataname:
#            chance = np.random.randint(100)
#            if chance < train_percentage:
#                train_datapath.append(datapath + path+ '/' + file)
#                train_label.append(i)
#            else:
#                test_datapath.append(datapath + path + '/' + file)
#                test_label.append(i)
#     print('train data:',len(train_datapath))
#     print('test data:',len(test_datapath))
#     return [train_datapath,train_label,test_datapath,test_label]



# def get_random_batch(train_datapath,train_label,batchsize,n_calss,img_width,img_height):
    
#     train_data = np.zeros([batchsize,img_width,img_height,3])
#     train_data =  train_data.astype(np.uint8)
#     train_label_onehot = np.zeros([batchsize,n_calss])
    
#     l = len(train_datapath)
#     i = 0
#     for _ in range(batchsize):
#         image_index = random.randrange(l)
#         img = cv2.imread(train_datapath[image_index])
#         train_data[i,:,:,:]  = cv2.resize(img,(img_height,img_width))
#         train_label_onehot[i,int(train_label[image_index])] = 1
# #        print(i,image_index,train_datapath[image_index])
#         i += 1
#     return train_data,train_label_onehot



# def get_test_data(test_datapath,test_label,n_calss,img_width,img_height):
#     test_data = np.zeros([len(test_datapath),img_width,img_height,3])
#     test_data = test_data.astype(np.uint8)
#     test_label_onehot = np.zeros([len(test_datapath),n_calss])
#     i = 0
#     for path in test_datapath:
#        img = cv2.imread(path)
#        test_data[i,:,:,:]  = cv2.resize(img,(img_height,img_width))
#        test_label_onehot[i,test_label[i]] = 1 
#        i += 1
#     return test_data,test_label_onehot
    
    
# train_datapath,train_label,test_datapath,test_label = get_datalist(datapath,95)

# train_data,train_label_onehot = get_random_batch(train_datapath,train_label,256,n_calss,img_width,img_height)


# test_data,test_label_onehot = get_test_data(test_datapath,test_label,n_calss,img_width,img_height)

import os
import cv2
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf

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
    # train_data =  train_data.astype(np.uint8)
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
    # test_data = test_data.astype(np.uint8)
    test_label_onehot = np.zeros([len(test_datapath),n_calss])
    i = 0
    for path in test_datapath:
       img = cv2.imread(path)
       test_data[i,:,:,:]  = cv2.resize(img,(img_height,img_width))
       test_label_onehot[i,test_label[i]] = 1 
       i += 1
    return test_data,test_label_onehot


def VGG():
    vgg16_npy_path = './model/vgg16.npy'
    vgg_mean = [103.939, 116.779, 123.68]

    data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

    def max_pool(bottom, name):
            return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, data_dict[name][1]))
            return lout

    def fc_layer(bottom,name):
        return tf.nn.bias_add(tf.matmul(bottom, data_dict[name][0]), data_dict[name][1])


            
    with tf.variable_scope('input'):        
        tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        tfy = tf.placeholder(tf.float32, [None, 5])
        
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=tfx * 255.0)
        # red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=tfx )

        bgr = tf.concat(axis=3, values=[
            blue - vgg_mean[0],
            green - vgg_mean[1],
            red - vgg_mean[2],
        ])
        # bgr = tf.concat(axis=3, values=[
        #     blue,
        #     green,
        #     red,
        # ])
        
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

    flatten = tf.reshape(pool5, [-1, 7*7*512])
    # fc6 = fc_layer(flatten, "fc6")
    fc6 = tf.nn.relu(fc_layer(flatten, "fc6"))


    return fc6,tfx
# flatten = tf.reshape(pool5, [-1, 4*4*512])

# fc6 = tf.layers.dense(flatten, 256, tf.nn.relu, name='fc6')
# out = tf.layers.dense(fc6, 5, tf.nn.softmax, name='out')

# out = tf.layers.dense(flatten,5,name='out')

# with tf.name_scope('final_training_ops'):
#     weights = tf.Variable(
#         tf.truncated_normal(
#             [4*4*512, 5], stddev=0.1))
#     biases = tf.Variable(tf.zeros([5]))
#     logits = tf.matmul(flatten, weights) + biases
#     out = tf.nn.softmax(logits)
img_width = 224
img_height = 224
train_percentage = 95
# train_datapath,train_label,test_datapath,test_label = get_datalist(datapath,train_percentage)
# test_data,test_label_onehot = get_test_data(test_datapath,test_label,n_calss,img_width,img_height)
path = './data/daisy/5547758_eea9edfd54_n.jpg'
img = skimage.io.imread(path)
# print(img)
img = img / 255.0
# print "Original Image Shape: ", img.shape
# we crop image from center
short_edge = min(img.shape[:2])
yy = int((img.shape[0] - short_edge) / 2)
xx = int((img.shape[1] - short_edge) / 2)
crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
# resize to 224, 224
resized_img = skimage.transform.resize(crop_img, (224, 224))[None, :, :, :]
# print(resized_img)

def test(sess,resized_img):
    bottom,tfx = VGG()
    bottleneck_values = sess.run(bottom, {tfx:resized_img})
    return bottleneck_values


with tf.Session() as sess:
    # bottom,tfx = VGG()
    # bottleneck_values = sess.run(bottom, {tfx:resized_img})
    bottleneck_values = test(sess,resized_img)
    print(bottleneck_values.shape)
    print(bottleneck_values[0].shape)
    bottleneck_path = './bottleneck/data.txt'
    # print(test_data.shape)
    bottleneck_string = ','.join(str(x) for x in bottleneck_values[0])
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)