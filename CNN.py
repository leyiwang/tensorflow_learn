import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)

def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)

def conv2d(x, W):
    '''
    Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
    horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
    '''
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")
#def gen_conv2d(patch_size, in_size, out_size, )
def accuracy(xs, y_):
    y_pre = sess.run(prediction, feed_dict={x: xs, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_, axis=1), tf.argmax(y_pre, axis=1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return sess.run(acc)

#define placeholder
with tf.name_scope(name="inputs"):
    keep_prob = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x_input")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_input")
    x_image = tf.reshape(x, [-1, 28, 28 , 1])#samp_num, length, weight, channel(height)

##conv1 layer##
W_conv1 = weight_variable([5, 5, 1, 32])#patch size5*5, in_size, out_size
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#out size 28*28*32
h_pool1 = max_pool_2x2(h_conv1)#out size 14*14*32
##conv2 layer##
W_conv2 = weight_variable([5, 5, 32, 64])#patch_size 5*5, in_size, out_size
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#out_size 14*14*64
h_pool2 = max_pool_2x2(h_conv2)#out_size 7*7*64
##func1 layer##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
##func2 layer##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope(name="loss"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(prediction), reduction_indices=[1]))

with tf.name_scope(name="optimization"):
    optimization = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
init = tf.global_variables_initializer()
tf.summary.scalar('cost', cross_entropy)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter("logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("logs/test", sess.graph)
    for step in range(5000):
        train_X, train_y = mnist.train.next_batch(100)
        _, cost = sess.run([optimization, cross_entropy], feed_dict={x:train_X, y_:train_y, keep_prob:0.5})
        if step % 50 == 0:
            train_result = sess.run(merged, feed_dict={x:train_X, y_:train_y, keep_prob: 1})
            test_result = sess.run(merged, feed_dict={x:mnist.test.images, y_: mnist.test.labels, keep_prob: 1})
            train_writer.add_summary(train_result, step)
            test_writer.add_summary(test_result, step)
            print(step, '=> train_acc:', accuracy(train_X, train_y), 'test_acc:', accuracy(mnist.test.images, mnist.test.labels))