import os
import math
import torch

from PIL import Image
from mhp_utils import mhp_image_list
from torch.utils.data import Dataset, DataLoader


class MHPSegmentation(Dataset):
	"""https://lv-mhp.github.io/dataset"""

	def __init__(self, root_dir, image_set='train', transforms=None):
		"""
		Parameters:
			root_dir (string): Root directory of the extracted LV-MHP-V2 dataset.
			image_set (string, optional): Select the image_set to use, ``train``, ``val``
			transforms (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir = root_dir
		self.image_set = image_set
		self.transforms = transforms

		self.images = []
		self.targets = []

		img_list = mhp_image_list(os.path.join(root_dir, 'list/{:s}.txt'.format(image_set)))

		for img_index in img_list:
			img_filename = os.path.join(root_dir, image_set, 'images/{:d}.jpg'.format(img_index))
			target_filename = os.path.join(root_dir, image_set, 'parsing_annos/{:d}.png'.format(img_index))

			if os.path.isfile(img_filename) and os.path.isfile(target_filename):
				self.images.append(img_filename)
				self.targets.append(target_filename)
		
	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		image = Image.open(self.images[index]).convert('RGB')
		target = Image.open(self.targets[index])

		if self.transforms is not None:
			image, target = self.transforms(image, target)

		return image, target

