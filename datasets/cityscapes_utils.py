import os
import copy
import torch
import torch.utils.data
import torchvision

from PIL import Image


#
# note:  is is slow to remap the label categories at runtime,
#        so use cityscapes_remap.py to do it in advance.
#
class FilterAndRemapCityscapesCategories(object):
    def __init__(self, categories, classes):
        self.categories = categories
        self.classes = classes
        print self.classes

    def __call__(self, image, anno):

       anno = copy.deepcopy(anno)
       for y in range(anno.height):
           for x in range(anno.width):
               org_label = anno.getpixel((x,y))

               if org_label not in self.categories:
                   anno.putpixel((x,y), 0)
        
       return image, anno


def get_cityscapes(root, image_set, transforms):

    #CAT_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    #transforms = Compose([
    #    FilterAndRemapCityscapesCategories(CAT_LIST, torchvision.datasets.Cityscapes.classes),
    #    transforms
    #])

    dataset = torchvision.datasets.Cityscapes(root, split=image_set, mode='fine', target_type='semantic', 
                                              transform=transforms, target_transform=transforms)

    return dataset
