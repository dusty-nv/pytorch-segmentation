from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier=None, export_onnx=False):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        self.export_onnx = export_onnx

        print('torchvision.models.segmentation.FCN() => configuring model for ' + ('ONNX export' if export_onnx else 'training'))


    def forward(self, x):
        input_shape = x.shape[-2:]

        # contract: features is a dict of tensors
        features = self.backbone(x)
        x = features["out"]
        x = self.classifier(x)

        # TensorRT doesn't support bilinear upsample, so when exporting to ONNX,
        # use nearest-neighbor upsampling, and also return a tensor (not an OrderedDict)
        if self.export_onnx:
            print('FCN configured for export to ONNX')
            print('FCN model input size = ' + str(input_shape))
            print('FCN classifier output size = ' + str(x.size()))
   
            #x = F.interpolate(x, size=(int(input_shape[0]), int(input_shape[1])), mode='nearest')

            print('FCN upsample() output size = ' + str(x.size()))
            print('FCN => returning tensor instead of OrderedDict')
            return x

        # non-ONNX training/eval path
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        result = OrderedDict()
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result
