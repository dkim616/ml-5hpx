import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import png
import numpy

def main():
	# Training, validation, and testing sets
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	# Create session
	sess = tf.InteractiveSession()

	print 'STARTING SIMPLE SOFTMAX'

	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])

	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))

	sess.run(tf.global_variables_initializer())

	y = tf.matmul(x,W) + b
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	for _ in range(1000):
		batch = mnist.train.next_batch(100)
		train_step.run(feed_dict={
			x: batch[0], 
			y_: batch[1]
		})

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print accuracy.eval(feed_dict={
		x: mnist.test.images,
		y_: mnist.test.labels
	})

	selected_two_images = []
	index = 0
	num_test_examples = mnist.test.num_examples
	while index < num_test_examples and len(selected_two_images) < 10:
		if mnist.test.labels[index][2] == 1:
			selected_two_images.append(mnist.test.images[index])
		index += 1

	print len(selected_two_images)

	for i, image in enumerate(selected_two_images):
		image = numpy.multiply(image, 255.0)
		image = image.astype(numpy.uint8)
		png.from_array(image.reshape(28, 28), 'L').save('image'+str(i)+'.png')

	print 'num images', len(mnist.test.images)
	print 'num labels', len(mnist.test.labels)

	sess.close()

if __name__ == '__main__':
	main()
