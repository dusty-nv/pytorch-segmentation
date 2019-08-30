#
# this script remaps the original DeepScene friburg_forest dataset annotations
# from RGB images (with 6 classes) to single-channel index images (with 5 classes).
#
# the original dataset can be downloaded from:  http://deepscene.cs.uni-freiburg.de/
#
import os
import copy
import argparse

from PIL import Image
from multiprocessing import Pool as ProcessPool


#
# map of existing class label ID's and colors to new ID's
# each entry consists of a tuple (new_ID, name, color)
#
CLASS_MAP = [ (0, 'trail', (170, 170, 170)),
              (1, 'grass', (0, 255, 0)),
		    (2, 'vegetation', (102, 102, 51)),
		    (3, 'obstacle', (0, 0, 0)),
		    (4, 'sky', (0, 120, 255)),
		    (2, 'void', (0, 60, 0)) ]	# 'void' appears to be trees in the dataset, so it is mapped to vegetation
			            

def lookup_class(color):
	for c in CLASS_MAP:
		if color == c[2]:
			return c[0]

	print('could not find class with color ' + str(color))
	return -1


def remap_labels(args):
	input_path = args[0]
	output_path = args[1]
	colorized = args[2]

	print('{:s} -> {:s}'.format(input_path, output_path))

	if os.path.isfile(output_path):
		print('skipping image {:s}, already exists'.format(output_path))
		return

	img_input = Image.open(input_path)
	img_output = Image.new('RGB' if colorized is True else 'L', (img_input.width, img_input.height))

	for y in range(img_input.height):
		for x in range(img_input.width):
			org_label = img_input.getpixel((x,y))
			new_label = CLASS_MAP[lookup_class(org_label)][2 if colorized else 0]
			img_output.putpixel((x,y), new_label)

	img_output.save(output_path)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Remap DeepScene Segmentation Images')
	parser.add_argument('input', type=str, metavar='IN', help='path to directory of annotated images to remap')
	parser.add_argument('output', type=str, metavar='OUT', help='path to directory to save remaped annotation images')
	parser.add_argument('--colorized', action='store_true', help='output colorized segmentation maps (RGB)')
	parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
	args = parser.parse_args()

	if not os.path.exists(args.output):
		os.makedirs(args.output)

	files = os.listdir(args.input)
	worker_args = []

	for n in range(len(files)):
		worker_args.append((os.path.join(args.input, files[n]), os.path.join(args.output, files[n]), args.colorized))

	#for n in worker_args:
	#    remap_labels(n)

	with ProcessPool(processes=args.workers) as pool:
		pool.map(remap_labels, worker_args)

