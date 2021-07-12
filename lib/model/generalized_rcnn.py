# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform, model_parallel=None):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn.to(model_parallel[0])
        self.roi_heads = roi_heads.to(model_parallel[0])
        self.model_parallel = model_parallel


    def _od_move_helper(self, od, target_dev):
        li = []
        for k,v in od.items():
            li.append((k, v.to(target_dev)))
        return OrderedDict(li)

    def _target_move_helper(self, target, target_dev):
        if target is None:
            return None
        result = []
        for each in target:
            result.append({"boxes": each["boxes"].to(target_dev), "labels": each["labels"].to(target_dev)})
        return result

    def _loss_move_helper(self, loss_dict, target_dev):
        for k, v in loss_dict.items():
            loss_dict[k] = v.to(target_dev)
        return loss_dict

    def _proposal_move_helper(self, proposals, target_dev):
        li = []
        for each in proposals:
            li.append(each.to(target_dev))
        return li



    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals, proposal_losses = self.rpn(
            images, self._od_move_helper(features, self.model_parallel[0]), 
            self._target_move_helper(targets, self.model_parallel[0]))
        
        detections, detector_losses = self.roi_heads(
            self._od_move_helper(features, self.model_parallel[0]), 
            proposals,
            images.image_sizes,
            self._target_move_helper(targets, self.model_parallel[0]))
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections

