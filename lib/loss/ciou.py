import os
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

# from core.config import cfg
# import nn as mynn


def bbox_transform(deltas, weights):
    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    dw = torch.clamp(dw, max=np.log(1000. / 16.))
    dh = torch.clamp(dh, max=np.log(1000. / 16.))

    pred_ctr_x = dx
    pred_ctr_y = dy
    pred_w = torch.exp(dw)
    pred_h = torch.exp(dh)

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    return x1.view(-1), y1.view(-1), x2.view(-1), y2.view(-1)

def compute_diou(output, target, bbox_inside_weights, bbox_outside_weights,
                transform_weights=None):
    if transform_weights is None:
        transform_weights = (1., 1., 1., 1.)
    x1, y1, x2, y2 = bbox_transform(output, transform_weights)
    x1g, y1g, x2g, y2g = bbox_transform(target, transform_weights)

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(output)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) +1e-7
    d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
    u = d / c
    diouk = iouk - u
    iou_weights = bbox_inside_weights.view(-1, 4).mean(1) * bbox_outside_weights.view(-1, 4).mean(1)
    iouk = ((1 - iouk) * iou_weights).sum(0) / output.size(0)
    diouk = ((1 - diouk) * iou_weights).sum(0) / output.size(0)

    return iouk, diouk


def compute_ciou(output, target, transform_weights=None):
    if transform_weights is None:
        transform_weights = (1., 1., 1., 1.)

    x1, y1, x2, y2 = bbox_transform(output, transform_weights)
    x1g, y1g, x2g, y2g = bbox_transform(target, transform_weights)

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)
    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g

    x_center = (x2 + x1) / 2
    y_center = (y2 + y1) / 2
    x_center_g = (x1g + x2g) / 2
    y_center_g = (y1g + y2g) / 2

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(output)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) +1e-7
    d = ((x_center - x_center_g) ** 2) + ((y_center - y_center_g) ** 2)
    u = d / c
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w_gt/h_gt)-torch.atan(w_pred/h_pred)),2)
    with torch.no_grad():
        S = 1 - iouk
        alpha = v / (S + v)
    ciouk = iouk - (u + alpha * v)
    # iou_weights = bbox_inside_weights.view(-1, 4).mean(1) * bbox_outside_weights.view(-1, 4).mean(1)
    iou_weights = 1
    iouk = ((1 - iouk) * iou_weights).sum(0) / output.size(0)
    ciouk = ((1 - ciouk) * iou_weights).sum(0) / output.size(0)

    return ciouk

def compute_giou(output, target, bbox_inside_weights, bbox_outside_weights,
                transform_weights=None):
    if transform_weights is None:
        transform_weights = (1., 1., 1., 1.)

    x1, y1, x2, y2 = bbox_transform(output, transform_weights)
    x1g, y1g, x2g, y2g = bbox_transform(target, transform_weights)

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(output)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
    giouk = iouk - ((area_c - unionk) / area_c)
    iou_weights = bbox_inside_weights.view(-1, 4).mean(1) * bbox_outside_weights.view(-1, 4).mean(1)
    iouk = ((1 - iouk) * iou_weights).sum(0) / output.size(0)
    giouk = ((1 - giouk) * iou_weights).sum(0) / output.size(0)

    return iouk, giouk

def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, beta=1.0):
    """
    SmoothL1(x) = 0.5 * x^2 / beta      if |x| < beta
                  |x| - 0.5 * beta      otherwise.
    1 / N * sum_i alpha_out[i] * SmoothL1(alpha_in[i] * (y_hat[i] - y[i])).
    N is the number of batch elements in the input predictions
    """
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < beta).detach().float()
    in_loss_box = smoothL1_sign * 0.5 * torch.pow(in_box_diff, 2) / beta + \
                  (1 - smoothL1_sign) * (abs_in_box_diff - (0.5 * beta))
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    N = loss_box.size(0)  # batch size
    loss_box = loss_box.view(-1).sum(0) / N
    return loss_box


# def clip_gradient(model, clip_norm):
#     """Computes a gradient clipping coefficient based on gradient norm."""
#     totalnorm = 0
#     for p in model.parameters():
#         if p.requires_grad:
#             modulenorm = p.grad.data.norm()
#             totalnorm += modulenorm ** 2
#     totalnorm = np.sqrt(totalnorm)
#
#     norm = clip_norm / max(totalnorm, clip_norm)
#     for p in model.parameters():
#         if p.requires_grad:
#             p.grad.mul_(norm)
#
#
# def decay_learning_rate(optimizer, cur_lr, decay_rate):
#     """Decay learning rate"""
#     new_lr = cur_lr * decay_rate
#     # ratio = _get_lr_change_ratio(cur_lr, new_lr)
#     ratio = 1 / decay_rate
#     if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
#         logger.info('Changing learning rate %.6f -> %.6f', cur_lr, new_lr)
#     # Update learning rate, note that different parameter may have different learning rate
#     for param_group in optimizer.param_groups:
#         cur_lr = param_group['lr']
#         new_lr = decay_rate * param_group['lr']
#         param_group['lr'] = new_lr
#         if cfg.SOLVER.TYPE in ['SGD']:
#             if cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
#                     ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
#                 _CorrectMomentum(optimizer, param_group['params'], new_lr / cur_lr)
#
# def update_learning_rate(optimizer, cur_lr, new_lr):
#     """Update learning rate"""
#     if cur_lr != new_lr:
#         ratio = _get_lr_change_ratio(cur_lr, new_lr)
#         if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
#             logger.info('Changing learning rate %.6f -> %.6f', cur_lr, new_lr)
#         # Update learning rate, note that different parameter may have different learning rate
#         param_keys = []
#         for ind, param_group in enumerate(optimizer.param_groups):
#             if ind == 1 and cfg.SOLVER.BIAS_DOUBLE_LR:  # bias params
#                 param_group['lr'] = new_lr * 2
#             else:
#                 param_group['lr'] = new_lr
#             param_keys += param_group['params']
#         if cfg.SOLVER.TYPE in ['SGD'] and cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
#                 ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
#             _CorrectMomentum(optimizer, param_keys, new_lr / cur_lr)
#
#
# def _CorrectMomentum(optimizer, param_keys, correction):
#     """The MomentumSGDUpdate op implements the update V as
#         V := mu * V + lr * grad,
#     where mu is the momentum factor, lr is the learning rate, and grad is
#     the stochastic gradient. Since V is not defined independently of the
#     learning rate (as it should ideally be), when the learning rate is
#     changed we should scale the update history V in order to make it
#     compatible in scale with lr * grad.
#     """
#     logger.info('Scaling update history by %.6f (new lr / old lr)', correction)
#     for p_key in param_keys:
#         optimizer.state[p_key]['momentum_buffer'] *= correction
#
#
# def _get_lr_change_ratio(cur_lr, new_lr):
#     eps = 1e-10
#     ratio = np.max(
#         (new_lr / np.max((cur_lr, eps)), cur_lr / np.max((new_lr, eps)))
#     )
#     return ratio
#
#
# def affine_grid_gen(rois, input_size, grid_size):
#
#     rois = rois.detach()
#     x1 = rois[:, 1::4] / 16.0
#     y1 = rois[:, 2::4] / 16.0
#     x2 = rois[:, 3::4] / 16.0
#     y2 = rois[:, 4::4] / 16.0
#
#     height = input_size[0]
#     width = input_size[1]
#
#     zero = Variable(rois.data.new(rois.size(0), 1).zero_())
#     theta = torch.cat([\
#       (x2 - x1) / (width - 1),
#       zero,
#       (x1 + x2 - width + 1) / (width - 1),
#       zero,
#       (y2 - y1) / (height - 1),
#       (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)
#
#     grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))
#
#     return grid
#
#
# def save_ckpt(output_dir, args, model, optimizer):
#     """Save checkpoint"""
#     if args.no_save:
#         return
#     ckpt_dir = os.path.join(output_dir, 'ckpt')
#     if not os.path.exists(ckpt_dir):
#         os.makedirs(ckpt_dir)
#     save_name = os.path.join(ckpt_dir, 'model_{}_{}.pth'.format(args.epoch, args.step))
#     if isinstance(model, mynn.DataParallel):
#         model = model.module
#     # TODO: (maybe) Do not save redundant shared params
#     # model_state_dict = model.state_dict()
#     torch.save({
#         'epoch': args.epoch,
#         'step': args.step,
#         'iters_per_epoch': args.iters_per_epoch,
#         'model': model.state_dict(),
#         'optimizer': optimizer.state_dict()}, save_name)
#     logger.info('save model: %s', save_name)
#
#
# def load_ckpt(model, ckpt):
#     """Load checkpoint"""
#     mapping, _ = model.detectron_weight_mapping
#     state_dict = {}
#     for name in ckpt:
#         if mapping[name]:
#             state_dict[name] = ckpt[name]
#     model.load_state_dict(state_dict, strict=False)
#
#
# def get_group_gn(dim):
#     """
#     get number of groups used by GroupNorm, based on number of channels
#     """
#     dim_per_gp = cfg.GROUP_NORM.DIM_PER_GP
#     num_groups = cfg.GROUP_NORM.NUM_GROUPS
#
#     assert dim_per_gp == -1 or num_groups == -1, \
#         "GroupNorm: can only specify G or C/G."
#
#     if dim_per_gp > 0:
#         assert dim % dim_per_gp == 0
#         group_gn = dim // dim_per_gp
#     else:
#         assert dim % num_groups == 0
#         group_gn = num_groups
#     return group_gn