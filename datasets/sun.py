import os
import math
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class SunRGBDSegmentation(Dataset):
	"""http://rgbd.cs.princeton.edu/challenge.html"""
	"""https://github.com/ankurhanda/sunrgbd-meta-data"""

	def __init__(self, root_dir, image_set='train', train_extra=True, transforms=None):
		"""
		Parameters:
			root_dir (string): Root directory of the dumped NYU-Depth dataset.
			image_set (string, optional): Select the image_set to use, ``train``, ``val``
			train_extra (bool, optional): If True, use extra images during training
			transforms (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir = root_dir
		self.image_set = image_set
		self.transforms = transforms

		self.images = []
		self.targets = []

		if image_set == 'train':
			train_images, train_targets = self.gather_images(os.path.join(root_dir, 'SUNRGBD-train_images'),
											    	    os.path.join(root_dir, 'train21labels'))

			self.images.extend(train_images)
			self.targets.extend(train_targets)

			if train_extra:			
				extra_images, extra_targets = self.gather_images(os.path.join(root_dir, 'SUNRGBD-trainextra_images'),
											         os.path.join(root_dir, 'trainextra21labels'))

				self.images.extend(extra_images)
				self.targets.extend(extra_targets)

		elif image_set == 'val':
			val_images, val_targets = self.gather_images(os.path.join(root_dir, 'SUNRGBD-test_images'),
											     os.path.join(root_dir, 'test21labels'))

			self.images.extend(val_images)
			self.targets.extend(val_targets)

	def gather_images(self, images_path, labels_path, max_images=5500):
		image_files = []
		label_files = []

		for n in range(max_images):
			image_filename = os.path.join(images_path, 'img-{:06d}.jpg'.format(n))
			label_filename = os.path.join(labels_path, 'img-{:06d}.png'.format(n))

			if os.path.isfile(image_filename) and os.path.isfile(label_filename):
				image_files.append(image_filename)
				label_files.append(label_filename)

		return image_files, label_files

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		image = Image.open(self.images[index]).convert('RGB')
		target = Image.open(self.targets[index])

		if self.transforms is not None:
			image, target = self.transforms(image, target)

		return image, target

