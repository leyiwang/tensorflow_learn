import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X, y = digits.data, digits.target
y = LabelBinarizer().fit_transform(y)#transorm one hot
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.33, random_state=1)

def add_layer(inputs, input_size, output_size, layer_name, active_function=None, keep_prob=1.0):
    with tf.name_scope(name=layer_name):
        with tf.name_scope(name=layer_name + "_w"):
            weight = tf.Variable(tf.random_normal([input_size, output_size]), name='W') #Gauss distribution
        with tf.name_scope(name=layer_name + "_b"):
            biases = tf.Variable(tf.zeros([1, output_size]) + 0.1, name="b")
        with tf.name_scope(name=layer_name + "_wx_plus_b"):
            wx_plus_b = tf.add(tf.matmul(inputs, weight), biases, name= "wx_plus_b")
            wx_plus_b = tf.nn.dropout(wx_plus_b, keep_prob, name="dropout")
        if active_function:
            wx_plus_b = active_function(wx_plus_b)
    return wx_plus_b

def accuracy(xs, y_):
    y_pre = sess.run(prediction, feed_dict={x: xs, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_, axis=1), tf.argmax(y_pre, axis=1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return sess.run(acc)

#define placeholder
with tf.name_scope(name="inputs"):
    keep_prob = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, shape=[None, 64], name="x_input")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_input")

#add layer
layer1 = add_layer(x, 64, 50, 'hidden', active_function=tf.nn.tanh, keep_prob=keep_prob)
prediction = add_layer(layer1, 50, 10, 'output', tf.nn.softmax,  keep_prob=keep_prob)
with tf.name_scope(name="loss"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(prediction), reduction_indices=[1]))

with tf.name_scope(name="optimization"):
    optimization = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()
tf.summary.scalar('cost', cross_entropy)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter("logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("logs/test", sess.graph)
    for step in range(5000):
        _, cost = sess.run([optimization, cross_entropy], feed_dict={x:train_X, y_:train_y, keep_prob:0.5})
        if step % 50 == 0:
            train_result = sess.run(merged, feed_dict={x:train_X, y_:train_y, keep_prob: 1})
            test_result = sess.run(merged, feed_dict={x:test_X, y_:test_y, keep_prob: 1})
            train_writer.add_summary(train_result, step)
            test_writer.add_summary(test_result, step)
            print(step, '=> train_acc:', accuracy(train_X, train_y), 'test_acc:', accuracy(test_X, test_y))