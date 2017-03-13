import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import cnn_model
import cnn_utils

def train():
	# Training, validation, and testing sets
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	checkpoint = cnn_utils.get_model_saves_path()

	x, y_, keep_prob, y_conv, variables = cnn_model.get_model()

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	saver = tf.train.Saver(var_list=variables)

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	# saver.restore(sess, checkpoint)

	for i in range(20000):
		batch = mnist.train.next_batch(50)
		if i % 100 == 0:
			saver.save(sess, checkpoint)
			train_accuracy = accuracy.eval(feed_dict={
				x: batch[0],
				y_: batch[1],
				keep_prob: 1.0
			})
			print 'step %d, training accuracy %g' % (i, train_accuracy)
		train_step.run(feed_dict={
			x: batch[0],
			y_: batch[1],
			keep_prob: 0.5
		})

	print 'test accuracy %g' % accuracy.eval(feed_dict={
		x: mnist.test.images,
		y_: mnist.test.labels,
		keep_prob: 1.0
	})

	sess.close()

if __name__ == '__main__':
	train()