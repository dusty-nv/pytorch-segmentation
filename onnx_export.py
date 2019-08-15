#
# converts a saved PyTorch model to ONNX format
# 
import os
import argparse

import torch
import torchvision.models as models


# parse command line
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='model_best.pth', help="path to input PyTorch model (default: model_best.pth)")
parser.add_argument('--output', type=str, default='', help="desired path of converted ONNX model (default: <ARCH>.onnx)")
parser.add_argument('--model-dir', type=str, default='', help="directory to look for the input PyTorch model in, and export the converted ONNX model to (if --output doesn't specify a directory)")

opt = parser.parse_args() 
print(opt)

# format input model path
if opt.model_dir:
   opt.model_dir = os.path.expanduser(opt.model_dir)
   opt.input = os.path.join(opt.model_dir, opt.input)

# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('running on device ' + str(device))

# load the model checkpoint
print('loading checkpoint:  ' + opt.input)
checkpoint = torch.load(opt.input)

arch = checkpoint['arch']
num_classes = checkpoint['num_classes']

# create the model architecture
print('using model:  ' + arch)
print('num classes:  ' + str(num_classes))

model = models.segmentation.__dict__[arch](num_classes=num_classes,
                                           aux_loss=None,
                                           pretrained=False,
                                           export_onnx=True)
																 
# load the model weights
model.load_state_dict(checkpoint['model'])

model.to(device)
model.eval()

print(model)
print('')

# create example image data
resolution = checkpoint['resolution']
input = torch.ones((1, 3, resolution[0], resolution[1])).cuda()
print('input size:  {:d}x{:d}'.format(resolution[1], resolution[0]))

# format output model path
if not opt.output:
   opt.output = arch + '.onnx'

if opt.model_dir and opt.output.find('/') == -1 and opt.output.find('\\') == -1:
   opt.output = os.path.join(opt.model_dir, opt.output)

# export the model
input_names = [ "input_0" ]
output_names = [ "output_0" ]

print('exporting model to ONNX...')
torch.onnx.export(model, input, opt.output, verbose=True, input_names=input_names, output_names=output_names)
print('model exported to:  {:s}'.format(opt.output))


