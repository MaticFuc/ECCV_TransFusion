from bisect import bisect

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage.measurements import label


def get_transparentfussion_sample(net, numSteps, data):
    with torch.no_grad():
        x = data.cuda()
        _, h, w = x.shape
        x = x.unsqueeze(0)
        mask = torch.zeros((1, 1, h, w)).cuda()
        all_masks = torch.zeros((1, 1, h, w)).cuda()
        for i in reversed(range(1, numSteps)):

            x, mask = net.p_sample(
                x, mask, x.new_full((1,), i, dtype=torch.long), return_mask=True
            )
            mask = nn.Sigmoid()(mask)

            all_masks += mask

            mask = (mask > 0.5).type(torch.FloatTensor).cuda()
        all_masks /= numSteps

    return all_masks, x


def normalize(x):
    if x.shape[0] == 1:
        return x
    x[:3, :, :] = (x[:3, :, :] / 2) + 0.5
    x[:3, :, :] = x[:3, :, :].clip(0, 1)
    return x


def calculate_transfussion_results(net, loader, index):
    true = loader.__getitem__(index)["image"]
    img_size = true.shape[-1]
    mask, x = get_transparentfussion_sample(net, net.steps - 1, true)

    mask = mask.detach().cpu().numpy().squeeze()
    x = x.detach().cpu().numpy().squeeze()
    true = true.detach().cpu().numpy().squeeze()

    if len(x.shape) == 2:
        x = x[None, :, :]
        true = true[None, :, :]

    true = normalize(true)
    x = normalize(x)

    mask_disc = np.expand_dims(mask, 0)
    mask_recon = np.zeros((img_size, img_size))
    for i in range(true.shape[0]):
        mask_recon += (true[i, :, :] - x[i, :, :]) ** 2
    mask_recon /= true.shape[0]

    return x, mask_disc, mask_recon


def trapezoid(x, y, x_max=None):

    x = np.asarray(x)
    y = np.asarray(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))

    x = x[finite_mask]
    y = y[finite_mask]

    correction = 0.0
    if x_max is not None:
        if x_max not in x:

            ins = bisect(x, x_max)

            y_interp = y[ins - 1] + (
                (y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1])
            )
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def compute_pro(anomaly_maps, ground_truth_maps):
    structure = np.ones((3, 3), dtype=int)

    num_ok_pixels = 0
    num_gt_regions = 0

    shape = (len(anomaly_maps), anomaly_maps[0].shape[0], anomaly_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)

    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(ground_truth_maps):
        labeled, n_components = label(gt_map, structure)
        num_gt_regions += n_components
        ok_mask = labeled == 0
        num_ok_pixels_in_map = np.sum(ok_mask)
        num_ok_pixels += num_ok_pixels_in_map

        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1

        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for k in range(n_components):
            region_mask = labeled == (k + 1)
            region_size = np.sum(region_mask)
            pro_change[region_mask] = 1.0 / region_size

        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()

    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]

    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    anomaly_scores_sorted = anomaly_scores_flat
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    fp_changes_sorted = fp_changes_flat
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
    pro_changes_sorted = pro_changes_flat

    del sort_idxs

    np.cumsum(fp_changes_sorted, out=fp_changes_sorted)
    fp_changes_sorted = fp_changes_sorted.astype(np.float32, copy=False)
    np.divide(fp_changes_sorted, num_ok_pixels, out=fp_changes_sorted)
    fprs = fp_changes_sorted

    np.cumsum(pro_changes_sorted, out=pro_changes_sorted)
    np.divide(pro_changes_sorted, num_gt_regions, out=pro_changes_sorted)
    pros = pro_changes_sorted

    keep_mask = np.append(np.diff(anomaly_scores_sorted) != 0, np.True_)
    del anomaly_scores_sorted

    fprs = fprs[keep_mask]
    pros = pros[keep_mask]
    del keep_mask

    np.clip(fprs, a_min=None, a_max=1.0, out=fprs)
    np.clip(pros, a_min=None, a_max=1.0, out=pros)

    zero = np.array([0.0])
    one = np.array([1.0])

    all_fprs = np.concatenate((zero, fprs, one))
    all_pros = np.concatenate((zero, pros, one))
    integration_limit = 0.3
    au_pro = trapezoid(all_fprs, all_pros, x_max=integration_limit)
    au_pro /= integration_limit

    return au_pro
