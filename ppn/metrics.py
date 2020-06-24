# sips.metrics
# author: tilleyd

import numpy as np

def precision_recall(pred_points, gt_points, thr):
    # square threshold
    thr *= thr

    # calculate squared difference hits
    num_pred = len(pred_points)
    num_gt = len(gt_points)
    if num_pred == 0 or num_gt == 0:
        return 0.0, 0.0

    diff = np.zeros((num_pred, num_gt), np.float32)

    for j in range(0, num_pred):
        for k in range(0, num_gt):
            xdiff = pred_points[j][0] - gt_points[k][0]
            xdiff *= xdiff
            ydiff = pred_points[j][1] - gt_points[k][1]
            ydiff *= ydiff
            diff[j][k] = xdiff + ydiff

    diff_pr = diff.copy()

    # calculate recall
    recall_hits = 0
    for c in range(0, diff.shape[1]):
        col = diff[:, c]
        nearest = np.argmin(col)
        if col[nearest] < thr:
            recall_hits += 1
            diff[nearest, :].fill(float('inf'))

    # calculate precision
    prec_hits = 0
    for r in range(0, diff_pr.shape[0]):
        row = diff_pr[r, :]
        nearest = np.argmin(row)
        if row[nearest] < thr:
            prec_hits += 1
            diff_pr[:, nearest].fill(float('inf'))

    return prec_hits / num_pred, recall_hits / num_gt
