# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import network.net_practice1_conv as net

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
current_dir = os.getcwd()

try:
    os.makedirs(os.path.join(current_dir, 'models/practice1_conv'))
    # os.makedirs("/tmp/model-subset")
except:
    pass

xs = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
ys = tf.placeholder(dtype=tf.float32, shape=[None, 10])
x_image = tf.reshape(xs, shape=[-1, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32)

prediction = net.net_conv_layer3(inputs=x_image, keep_prob=keep_prob)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), 1))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10001):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob:0.5})
        if i % 1000 == 0:
            y_pre = sess.run(prediction, feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1})
            correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(mnist.test.labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            result = sess.run(accuracy, feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1})
            print result
    save_path = saver.save(sess, os.path.join(current_dir, 'models/practice1_conv/.ckpt'))