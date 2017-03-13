import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy

import cnn_model
import cnn_utils

def fool():
	# Training, validation, and testing sets
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	x, y_, keep_prob, y_conv, variables = cnn_model.get_model()

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	grad = tf.gradients(cross_entropy, x)

	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Create session
	sess = tf.InteractiveSession()
	saver = tf.train.Saver(variables)

	# sess.run(tf.global_variables_initializer())
	saver.restore(sess, cnn_utils.get_model_saves_path())

	two_image = None
	two_onehot = None
	index = 0
	num_test_examples = mnist.test.num_examples
	while index < num_test_examples and two_image is None:
		if mnist.test.labels[index][2] == 1:
			two_image = mnist.test.images[index].reshape(1, 784)
			two_onehot = mnist.test.labels[index].reshape(1, 10)
		index += 1

	six_onehot = None
	index = 0
	while index < num_test_examples and six_onehot is None:
		if mnist.test.labels[index][6] == 1:
			six_onehot = mnist.test.labels[index].reshape(1, 10)
		index += 1

	np_grad = sess.run(grad, feed_dict={
		x: two_image,
		y_: six_onehot,
		keep_prob: 1.0
	})
	signed_grad = numpy.sign(np_grad[0])
	delta_image_unsigned = 0.1 * np_grad[0]
	delta_image = 0.01 * signed_grad
	adv_image = delta_image + two_image

	print sess.run(y_conv, feed_dict={
		x: two_image,
		keep_prob: 1.0
	})

	print two_onehot
	print six_onehot

	print sess.run(y_conv, feed_dict={
		x: adv_image,
		keep_prob: 1.0
	})

	cnn_utils.save_image(np_grad[0], 'unsigned_grad')
	cnn_utils.save_image(signed_grad, 'signed_grad')
	cnn_utils.save_image(delta_image_unsigned, 'delta_image_unsigned')
	cnn_utils.save_image(delta_image, 'delta_image')
	cnn_utils.save_image(adv_image, 'adv_image')
	cnn_utils.save_image(two_image, 'two_image')

	print two_image
	print delta_image
	print adv_image

	sess.close()

if __name__ == '__main__':
	fool()
