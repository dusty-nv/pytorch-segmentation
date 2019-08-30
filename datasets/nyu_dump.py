#
# this script reads the .mat files from the NYU-Depth datasets
# and dumps their contents to individual image files for training
#
import os
import random
import argparse
import h5py
import numpy as np

from PIL import Image


parser = argparse.ArgumentParser(description='Dump NYU-Depth .mat files')
parser.add_argument('input', type=str, nargs='+', metavar='IN', help='paths to input .mat files')
parser.add_argument('--output', type=str, default='dump', metavar='OUT', help='path to directory to save dataset')
parser.add_argument("--images", action="store_true", help="dump RGB images")
parser.add_argument("--labels", action="store_true", help="dump label images")
parser.add_argument("--depth", action="store_true", help="dump depth images")
parser.add_argument("--depth-levels", type=int, default=20, help="number of disparity depth levels (default: 20)")
parser.add_argument("--split", action="store_true", help="dump train/val split files")
parser.add_argument("--split-val", type=float, default=0.15, help="fraction of dataset to split between train/val")

args = parser.parse_args()

input_images = []
input_labels = []
input_depths = []

global_depth_min = 1000.0
global_depth_max = -1000.0


#
# load arrays from .mat files
#
for filename in args.input:
	print('\n==> loading ' + filename)
	mat = h5py.File(filename, 'r')
	print(list(mat.keys()))

	if args.images or args.split:
		print('reading images')

		images = mat['images']
		images = np.array(images)

		print(images.shape)

		pixel_min = images.min()
		pixel_max = images.max()
		pixel_avg = images.mean()

		print('min pixel:  {:f}'.format(pixel_min))
		print('max pixel:  {:f}'.format(pixel_max))
		print('avg pixel:  {:f}'.format(pixel_avg))

		input_images.append(images)

	if args.depth:
		print('reading depths')

		depths = mat['depths']
		depths = np.array(depths)

		print(depths.shape)

		depth_min = depths.min()
		depth_max = depths.max()
		depth_avg = depths.mean()

		print('min depth:  {:f}'.format(depth_min))
		print('max depth:  {:f}'.format(depth_max))
		print('avg depth:  {:f}'.format(depth_avg))

		if depth_min < global_depth_min:
			global_depth_min = depth_min

		if depth_max > global_depth_max:
			global_depth_max = depth_max

		input_depths.append(depths)

print('')


#
# process source images
#
if args.images:
	images_path = os.path.join(args.output, 'images')

	if not os.path.exists(images_path):
		os.makedirs(images_path)

	for n in range(len(input_images)):
		for m in range(input_images[n].shape[0]):
			img_name = 'v{:d}_{:04d}.png'.format(n+1, m)
			img_path = os.path.join(images_path, img_name)

			print('processing image ' + img_path)

			img_in = input_images[n][m]
			img_in = np.moveaxis(img_in, [0, 1, 2], [2, 1, 0])
			#print(img_in.shape)

			img_out = Image.fromarray(img_in.astype('uint8'), 'RGB')
			print(img_out.size)
			img_out.save(img_path)
			

#
# process depth images
#
if args.depth:
	print('global min depth:  {:f}'.format(global_depth_min))
	print('global max depth:  {:f}'.format(global_depth_max))

	depth_path = os.path.join(args.output, 'depth')

	if not os.path.exists(depth_path):
		os.makedirs(depth_path)

	for n in range(len(input_depths)):
		print('\nprocessing depth/v{:d}'.format(n+1))

		arr = input_depths[n]

		# rescale the depths to lie between [0, args.depth_levels]
		arr = np.subtract(arr, global_depth_min)
		arr = np.multiply(arr, 1.0 / (global_depth_max - global_depth_min) * float(args.depth_levels))

		depth_min = arr.min()
		depth_max = arr.max()
		depth_avg = arr.mean()

		print('min depth:  {:f}'.format(depth_min))
		print('max depth:  {:f}'.format(depth_max))
		print('avg depth:  {:f}'.format(depth_avg))

		for m in range(arr.shape[0]):
			img_name = 'v{:d}_{:04d}.png'.format(n+1, m)
			img_path = os.path.join(depth_path, img_name)

			print('processing depth ' + img_path)

			img_in = arr[m]
			img_in = np.moveaxis(img_in, [0, 1], [1, 0])
			#print(img_in.shape)

			img_out = Image.fromarray(img_in.astype('uint8'), 'L')
			print(img_out.size)
			img_out.save(img_path)

#
# output train/val splits
#
if args.split:
	print('creating train/val splits')

	train_file = open(os.path.join(args.output, 'train.txt'), 'w')
	val_file = open(os.path.join(args.output, 'val.txt'), 'w')

	train_count = 0
	val_count = 0

	for n in range(len(input_images)):
		for m in range(input_images[n].shape[0]):
			img_name = 'v{:d}_{:04d}.png'.format(n+1, m)
			rand = random.random()
			
			if rand < args.split_val:
				val_file.write(img_name + '\n')
				val_count = val_count + 1
			else:
				train_file.write(img_name + '\n')
				train_count = train_count + 1

	print('total: {:d}'.format(train_count + val_count))
	print('train: {:d}'.format(train_count))
	print('val:   {:d}'.format(val_count))

	train_file.close()
	val_file.close()

