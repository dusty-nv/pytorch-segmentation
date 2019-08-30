import os
import math
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class NYUDepth(Dataset):
	"""https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html"""

	def __init__(self, root_dir, image_set='train', transforms=None):
		"""
		Parameters:
			root_dir (string): Root directory of the dumped NYU-Depth dataset.
			image_set (string, optional): Select the image_set to use, ``train``, ``val``
			transforms (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir = root_dir
		self.image_set = image_set
		self.transforms = transforms

		self.images = []
		self.targets = []

		img_list = self.read_image_list(os.path.join(root_dir, '{:s}.txt'.format(image_set)))

		for img_name in img_list:
			img_filename = os.path.join(root_dir, 'images/{:s}'.format(img_name))
			target_filename = os.path.join(root_dir, 'depth/{:s}'.format(img_name))

			if os.path.isfile(img_filename) and os.path.isfile(target_filename):
				self.images.append(img_filename)
				self.targets.append(target_filename)

	def read_image_list(self, filename):
		"""
		Read one of the image index lists

		Parameters:
			filename (string):  path to the image list file

		Returns:
			list (int):  list of strings that correspond to image names
		"""
		list_file = open(filename, 'r')
		img_list = []

		while True:
			next_line = list_file.readline()

			if not next_line:
				break

			img_list.append(next_line.rstrip())

		return img_list

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		image = Image.open(self.images[index]).convert('RGB')
		target = Image.open(self.targets[index])

		if self.transforms is not None:
			image, target = self.transforms(image, target)

		return image, target

