import tensorflow as tf

def Weights(shape, name=None):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), dtype=tf.float32, name=name)

def biases(shape, name=None):
    return tf.Variable(tf.constant(0.1, shape=shape), dtype=tf.float32, name=name)

def net_conv_layer3(inputs, keep_prob):
    # training step

    #layer1
    Weights_conv1 = tf.Variable(tf.truncated_normal(shape=[5,5,1,32], stddev=0.1), dtype=tf.float32, name='W_conv1')
    biases_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), dtype=tf.float32, name='b_conv1')
    conv_1 = tf.nn.relu(tf.nn.conv2d(inputs, Weights_conv1, strides=[1,1,1,1], padding='SAME') + biases_conv1)
    pool_layer1 = tf.nn.max_pool(conv_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  # outputs.shape = [n_samples, 14,14,32]
    conv_layer1 = pool_layer1

    #layer2
    Weights_conv2 = tf.Variable(tf.truncated_normal(shape=[5,5,32,48], stddev=0.1), dtype=tf.float32, name='W_conv2')
    biases_conv2 = tf.Variable(tf.constant(0.1, shape=[48]), dtype=tf.float32, name='b_conv2')
    conv_2 = tf.nn.relu(tf.nn.conv2d(pool_layer1, Weights_conv2, strides=[1,1,1,1], padding='SAME') + biases_conv2)  # outputs.shape = [n_samples, 14,14,48]
    # pool_layer2 = tf.nn.max_pool(conv_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv_layer2 = conv_2

    #layer3
    Weights_conv3 = tf.Variable(tf.truncated_normal(shape=[5,5,48,64], stddev=0.1), dtype=tf.float32, name='W_conv3')
    biases_conv3 = tf.Variable(tf.constant(0.1, shape=[64]), dtype=tf.float32, name='b_conv3')
    conv_3 = tf.nn.relu(tf.nn.conv2d(conv_2, Weights_conv3, strides=[1,1,1,1], padding='SAME') + biases_conv3)
    pool_layer3 = tf.nn.max_pool(conv_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  #outputs.shape = [n_samples, 7,7,64]
    conv_layer3 = pool_layer3

    #fulled layer1
    Weights_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 1024], stddev=0.1), dtype=tf.float32, name='W_fc1')
    biases_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), dtype=tf.float32, name='b_fc1')
    fc1 = tf.reshape(pool_layer3, shape=[-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(fc1, Weights_fc1) + biases_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    fc_layer1 = h_fc1

    #fulled layer2
    Weights_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.1), dtype=tf.float32, name='W_fc2')
    biases_fc2 = tf.Variable(tf.constant(0.1, shape=[10]), dtype=tf.float32, name='b_fc2')
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, Weights_fc2) + biases_fc2)

    return prediction
