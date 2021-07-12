'''
Minor modificatin from https://github.com/pytorch/vision/blob/2b73a4846773a670632b29fb2fc2ac57df7bce5d/torchvision/models/detection/backbone_utils.py#L43
'''
from collections import OrderedDict

from torch import nn
import torch.nn.functional as F
# from torchvision.models import resnet

from lib.model import res2net


class BackboneWithFasterRCNN(nn.Module):

    def __init__(self, backbone, model_parallel):
        super(BackboneWithFasterRCNN, self).__init__()
        self.part1 = nn.Sequential(
            OrderedDict([
                ['conv1', backbone.conv1],
                ['bn1', backbone.bn1],
                ['relu', backbone.relu],
                ['maxpool', backbone.maxpool],
                ['layer1', backbone.layer1],  # res2
            ])
        ).to(model_parallel[0])

        self.part2 = nn.Sequential(
            OrderedDict([
                ['layer2', backbone.layer2],  # res3
                ['layer3', backbone.layer3]  # res4
            ])
        ).to(model_parallel[1])

        self.out_channels = 1024
        self.model_parallel = model_parallel

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = self.part2(self.part1(x).to(self.model_parallel[1]))
        return OrderedDict([['feat_res4', feat]])


class RCNNConvHead(nn.Sequential):
    """docstring for RCNNConvHead"""

    def __init__(self, backbone):
        super(RCNNConvHead, self).__init__(
            OrderedDict(
                [['layer4', backbone.layer4]]  # res5
            )
        )
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = super(RCNNConvHead, self).forward(x)
        return OrderedDict([
            # Global average pooling
            ['feat_res4', F.adaptive_max_pool2d(x, 1)],
            ['feat_res5', F.adaptive_max_pool2d(feat, 1)]]
        )


def res2net_backbone(backbone_name, pretrained, model_parallel=None):
    backbone = res2net.__dict__[backbone_name](
        pretrained=pretrained)

    # freeze layers
    # backbone.conv1.weight.requires_grad_(False)
    # backbone.bn1.weight.requires_grad_(False)
    # backbone.bn1.bias.requires_grad_(False)

    stem = BackboneWithFasterRCNN(backbone, model_parallel)
    head = RCNNConvHead(backbone)

    return stem, head
