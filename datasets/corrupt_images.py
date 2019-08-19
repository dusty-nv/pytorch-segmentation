#
# this script detects corrupted images in a directory,
# and optionally moves them to a specified directory
#
import argparse
import warnings
import shutil

from os import listdir
from os import remove
from os.path import join

from PIL import Image

parser = argparse.ArgumentParser(description='corrupt image remover')
parser.add_argument('dir', metavar='DIR', help='path to directory')
parser.add_argument('--move', type=str, default=None, help='optional path to directory that corrupt images are moved to')
args = parser.parse_args()

num_bad = 0

warnings.filterwarnings("error")


for filename in listdir(args.dir):
  file_low = filename.lower()
  if file_low.endswith('.png') or file_low.endswith('.jpg') or file_low.endswith('.jpeg') or file_low.endswith('.gif'):
    file_path = join(args.dir,filename)
    try:
      #img = Image.open(file_path) # open the image file
      #img.verify() # verify that it is, in fact an image

      img = Image.open(file_path)
      img.load()

      imgRGB = img.convert('RGB')

      #if img.width < 16 or img.height < 16:
      #      print('Strange image dimensions ({:d}x{:d}): {:s}'.format(img.width, img.height, file_path))

      if img.width < 16 or img.height < 16:
            print('Bad image dimensions ({:d}x{:d}): {:s}'.format(img.width, img.height, file_path)) # print out the names of corrupt files
            
            if args.move is not None:
                  shutil.move(file_path, args.move) #remove(file_path)

            num_bad = num_bad + 1   
 
    except (IOError, SyntaxError, UserWarning, RuntimeWarning) as e:
      print('Bad image: {:s}'.format(file_path)) # print out the names of corrupt files

      if args.move is not None:
           shutil.move(file_path, args.move) #remove(file_path)

      num_bad = num_bad + 1

print('Detected {:d} corrupted images from {:s} '.format(num_bad, args.dir))
