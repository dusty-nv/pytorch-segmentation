import os

def mhp_image_list(filename):
	"""
	Read one of the image index lists from LV-MHP-v2/list

	Parameters:
		filename (string):  path to the image list file

	Returns:
		list (int):  list of int's that correspond to image names
	"""
	list_file = open(filename, 'r')
	img_list = []

	while True:
		next_line = list_file.readline()

		if not next_line:
			break

		img_list.append(int(next_line))

	return img_list

