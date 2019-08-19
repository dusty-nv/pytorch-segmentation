#
# this script remaps the original MHP class label ID's (range 0-58)
# to lie within a range that the segmentation networks support (21 classes)
#
# see below for the mapping of the original class ID's to the new class ID's,
# along with the class descriptions, some of which are combined from the originals.
#
# this script will overwrite the *_labelIds.png files from the gtCoarse/gtFine sets.
# to run it, launch these example commands for the desired train/train_extra/val sets:
#
#     $ python3 mhp_remap.py <path-to-cityscapes>/gtCoarse/train
#     $ python3 mhp_remap.py <path-to-cityscapes>/gtCoarse/val
#   
import os
import copy
import argparse

from PIL import Image
from mhp_utils import mhp_image_list
from multiprocessing import Pool as ProcessPool


#
# map of existing class label ID's (range 0-58) to new ID's (range 0-21)
#
LABEL_MAP = [0,  # Background
             1,  # Cap/hat
             1,  # Helmet
             2,  # Face
             3,  # Hair
             4,  # Left-arm
             4,  # Right-arm
             5,  # Left-hand
             5,  # Right-hand
             19, # Protector
             9,  # Bikini/bra
             7,  # Jacket/windbreaker/hoodie
             6,  # Tee-shirt
             6,  # Polo-shirt
             6,  # Sweater
             8,  # Singlet
             10, # Torso-skin
             11, # Pants
             12, # Shorts/swim-shorts
             12, # Skirt
             13, # Stockings
             13, # Socks
             14, # Left-boot
             14, # Right-boot
             14, # Left-shoe
             14, # Right-shoe
             14, # Left-highheel
             14, # Right-highheel
             14, # Left-sandal
             14, # Right-sandal
             15, # Left-leg
             15, # Right-leg
             16, # Left-foot
             16, # Right-foot
             7,  # Coat
             8,  # Dress
             8,  # Robe
             8,  # Jumpsuit
             8,  # Other-full-body-clothes
             1,  # Headwear
             17, # Backpack
             20, # Ball
             20, # Bats
             19, # Belt
             20, # Bottle
             17, # Carrybag
             17, # Cases
             18, # Sunglasses
             18, # Eyewear
             19, # Glove
             19, # Scarf
             20, # Umbrella
             17, # Wallet/purse
             19, # Watch
             19, # Wristband
             19, # Tie
             19, # Other-accessory
             6,  # Other-upper-body-clothes
             11] # Other-lower-body-clothes              

#
# new class label names, corresponding to remapped class ID's (range 0-21)
#
"""
void
hat/helmet/headwear
face
hair
arm
hand
shirt
jacket/coat
dress/robe
bikini/bra
torso_skin
pants
shorts
socks/stockings
shoe/boot
leg
foot
backpack/purse/bag
sunglasses/eyewear
other_accessory
other_item
"""

def remap_labels(args):
    input_dir = args[0]
    output_dir = args[1]
    img_index = args[2]

    src_images = 0
    img_output = None

    # check if this image has already been processed (i.e. by a previous run)
    output_path = os.path.join(output_dir, '{:d}.png'.format(img_index))

    if os.path.isfile(output_path):
        print('skipping image {:d}, already exists'.format(img_index))
        return

    # determine the number of source images for this frame
    for n in range(1,30):
        if os.path.isfile(os.path.join(input_dir, '{:d}_{:02d}_01.png'.format(img_index, n))):
            src_images = n

    print('processing image {:d} \t(src_images={:d})'.format(img_index, src_images))

    # aggregate and remap the source images into one output
    for n in range(1, src_images+1):
        img_input = Image.open(os.path.join(input_dir, '{:d}_{:02d}_{:02d}.png'.format(img_index, src_images, n)))
        
        if img_output is None:
            img_output = Image.new('L', (img_input.width, img_input.height))

        for y in range(img_input.height):
            for x in range(img_input.width):
                org_label = img_input.getpixel((x,y))[0]
                new_label = LABEL_MAP[org_label]

                if new_label != 0:   # only overwrite non-background pixels
                    img_output.putpixel((x,y), new_label)

                #if org_label != 0: 
                #    print('img {:d}_{:02d} ({:d}, {:d})  {:d} -> {:d}'.format(img_index, n, x, y, org_label, new_label))

    img_output.save(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Remap MHP Annotation Images')
    parser.add_argument('input', type=str, metavar='IN', help='path to directory of annotated images to remap')
    parser.add_argument('output', type=str, metavar='OUT', help='path to directory to save remaped annotation images')
    parser.add_argument('--list', type=str, required=True, metavar='LIST', help='path to image list')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    img_list = mhp_image_list(args.list)
    pool_args = []

    for img_index in img_list:
        pool_args.append( (args.input, args.output, img_index) )

    with ProcessPool(processes=args.workers) as pool:
        pool.map(remap_labels, pool_args)

