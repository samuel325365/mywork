# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print type(mnist)
print mnist.train.num_examples
print mnist.validation.num_examples
print mnist.test.num_examples

print("讓我們看一下 MNIST 訓練還有測試的資料集長得如何")
train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels
print
print " train_img 的 type : %s" % (type(train_img))
print " train_img 的 dimension : %s" % (train_img.shape,)
print " train_label 的 type : %s" % (type(train_label))
print " train_label 的 dimension : %s" % (train_label.shape,)
print " test_img 的 type : %s" % (type(test_img))
print " test_img 的 dimension : %s" % (test_img.shape,)
print " test_label 的 type : %s" % (type(test_label))
print " test_label 的 dimension : %s" % (test_label.shape,)

trainimg = mnist.train.images
trainlabel = mnist.train.labels

randidx = np.random.randint(trainimg.shape[0], size=1)


for i in [10,1,2]:
    curr_img = np.reshape(trainimg[i, :], (28,28))
    curr_label = np.argmax(trainlabel[i, :])
    print curr_label
    plt.matshow(curr_img, cmap='gray')
    plt.show()