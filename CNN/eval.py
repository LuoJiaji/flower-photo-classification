# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 10:19:00 2018

@author: Bllue
"""
import tensorflow as tf
import numpy as np
import os 
import cv2

filename = os.listdir('data/')

CHECKPOINT_DIR = './ckpt'
train_percentage = 90
img_width = 100
img_height = 100


train_datapath = []
train_label = []
test_datapath = []
test_label = []

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
    
    

checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
print(checkpoint_file)

with tf.Graph().as_default() as graph:
    with tf.Session().as_default() as sess:
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
    # 这里可以遍历模型各个节点
    #    for op in tf.get_default_graph().get_operations():
    #        print(op.name)
    #        print(op.values())
        input_x = graph.get_operation_by_name('input/Placeholder').outputs[0]
        dropout =  graph.get_operation_by_name('dropout/Placeholder').outputs[0]
        predictions = graph.get_operation_by_name('acc/ArgMax').outputs[0]
        
        all_predictions = sess.run(predictions, {input_x: test_data,dropout:1})
        
        correct = tf.equal(all_predictions, (tf.argmax(test_label_onehot,1)))
        acc = tf.reduce_mean(tf.cast(correct,tf.float32))
        print(sess.run(acc))
