import png

png.from_array([[255, 0, 0, 255],
				[0, 255, 255, 0]], 'L').save('test_image.png')