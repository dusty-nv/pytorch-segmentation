import os
import re
import math
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class DeepSceneSegmentation(Dataset):
	"""http://deepscene.cs.uni-freiburg.de/"""

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
			train_images, train_targets = self.gather_images(os.path.join(root_dir, 'train/rgb'),
											    	    os.path.join(root_dir, 'train/GT_index'))

			self.images.extend(train_images)
			self.targets.extend(train_targets)

			if train_extra:			
				extra_images, extra_targets = self.gather_images(os.path.join(root_dir, 'trainextra/rgb'),
											         	    os.path.join(root_dir, 'trainextra/GT_index'))

				self.images.extend(extra_images)
				self.targets.extend(extra_targets)

		elif image_set == 'val':
			val_images, val_targets = self.gather_images(os.path.join(root_dir, 'test/rgb'),
											     os.path.join(root_dir, 'test/GT_index'))

			self.images.extend(val_images)
			self.targets.extend(val_targets)

	def gather_images(self, images_path, labels_path):
		def sorted_alphanumeric(data):
			convert = lambda text: int(text) if text.isdigit() else text.lower()
			alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
			return sorted(data, key=alphanum_key)

		image_files = sorted_alphanumeric(os.listdir(images_path))
		label_files = sorted_alphanumeric(os.listdir(labels_path))

		if len(image_files) != len(label_files):
			print('warning:  images path has a different number of files than labels path')
			print('   ({:d} files) - {:s}'.format(len(image_files), images_path))
			print('   ({:d} files) - {:s}'.format(len(label_files), labels_path))
			
		for n in range(len(image_files)):
			image_files[n] = os.path.join(images_path, image_files[n])
			label_files[n] = os.path.join(labels_path, label_files[n])
			
			#print('{:s} -> {:s}'.format(image_files[n], label_files[n]))

		return image_files, label_files

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		image = Image.open(self.images[index]).convert('RGB')
		target = Image.open(self.targets[index])

		if self.transforms is not None:
			image, target = self.transforms(image, target)

		return image, target

