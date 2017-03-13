import os

import numpy
import png

def get_model_saves_path():
	current_file = os.path.realpath(__file__)
	current_file = current_file.split('/')
	current_file = '/'.join(current_file[:-1])
	current_file += '/model_saves/cnn_variables.checkpoint'
	return current_file

def save_image(image, name=None):
	if image is None:
		print 'Did not save image', name
		return
	if name is None:
		name = 'temp'
	image = numpy.multiply(image, 255.0)
	image = image.astype(numpy.uint8)
	png.from_array(image.reshape(28, 28), 'L').save(name + '.png')