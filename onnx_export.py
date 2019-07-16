#
# converts a saved PyTorch model to ONNX format
# 
import os
import argparse

import torch
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# parse command line
parser = argparse.ArgumentParser()
#parser.add_argument('--input', type=str, default='model_best.pth.tar', help="path to input PyTorch model (default: model_best.pth.tar)")
parser.add_argument('--output', type=str, default='', help="desired path of converted ONNX model (default: <ARCH>.onnx)")
#parser.add_argument('--model-dir', type=str, default='', help="directory to look for the input PyTorch model in, and export the converted ONNX model to (if --output doesn't specify a directory)")
#parser.add_argument('--no-softmax', action='store_true', help="disable adding nn.Softmax layer to model (default is to add Softmax)")

opt = parser.parse_args() 
print(opt)

# format input model path
#if opt.model_dir:
#	opt.model_dir = os.path.expanduser(opt.model_dir)
#	opt.input = os.path.join(opt.model_dir, opt.input)

# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('running on device ' + str(device))

# load the model checkpoint
#print('loading checkpoint:  ' + opt.input)
#checkpoint = torch.load(opt.input)
#arch = checkpoint['arch']

# create the model architecture
#print('using model:  ' + arch)
model = models.segmentation.__dict__['fcn_resnet101'](num_classes=21,
                                                      aux_loss=False,
                                                      pretrained=True)
																 
# load the model weights
#model.load_state_dict(checkpoint['state_dict'])

model.to(device)
model.eval()

print(model)

# create example image data
resolution = 480
#resolution = checkpoint['resolution']
input = torch.ones((1, 3, resolution, resolution)).cuda()
print('input size:  {:d}x{:d}'.format(resolution, resolution))

# format output model path
#if not opt.output:
#	opt.output = arch + '.onnx'

#if opt.model_dir and opt.output.find('/') == -1 and opt.output.find('\\') == -1:
#	opt.output = os.path.join(opt.model_dir, opt.output)

# export the model
input_names = [ "input_0" ]
output_names = [ "output_0" ]

print('exporting model to ONNX...')
torch.onnx.export(model, input, opt.output, verbose=True, input_names=input_names, output_names=output_names)
print('model exported to:  {:s}'.format(opt.output))


