import input_data
import tensorflow as tf

print 'Loading data...'
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print 'Loading data done!'

"""Data set"""
print
print 'Test data: ' + str(len(mnist.test.labels))
print 'Train data: ' + str(len(mnist.train.labels))
print 'Validation data: ' + str(len(mnist.validation.labels))
print

"""Net"""
print
print 'Number of input: ' + str(784)
print 'Using bias: ' + str(True)
print 'Number of hidden layer: ' + str(0)
print 'Number of hidden units: ' + str(0)
print 'Number of outputs ' + str(10)
print


"""Input"""
x = tf.placeholder("float", [None, 784])

"""Weights"""
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

"""Output"""
y = tf.nn.softmax(tf.matmul(x,W) + b)

"""Predicted output"""
y_ = tf.placeholder("float", [None,10])


"""Initialize"""
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

"""Train"""
print 'Start training...'
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print 'Training done!'
print

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})