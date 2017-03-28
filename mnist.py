import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs, input_size, output_size, active_function=None):
    weight = tf.Variable(tf.random_normal([input_size, output_size]))
    biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)
    wx_plus_b = tf.add(tf.matmul(inputs, weight), biases)
    if active_function:
        wx_plus_b = active_function(wx_plus_b)
    return wx_plus_b

def accuracy(xs, y_):
    y_pre = sess.run(prediction, feed_dict={x: xs})
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_pre, axis=1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return sess.run(acc)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

prediction = add_layer(x, 784, 10, active_function=tf.nn.softmax)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(prediction), reduction_indices=[1]))
optimization = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(5000):
        batch, label = mnist.train.next_batch(100)
        _, cost = sess.run([optimization, cross_entropy], feed_dict={x:batch, y_:label})
        if step % 50==0:
            print(step, '=> train_acc:', accuracy(batch, label), 'test_acc:', accuracy(mnist.test.images, mnist.test.labels))