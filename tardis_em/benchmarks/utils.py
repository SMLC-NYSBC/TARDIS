#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.contrib import tzip


def benchmark_cnn(logits: list, targets: list, reduce="mean", tqdm_pb=True):
    all_precisions = []
    all_recall = []
    all_f1 = []

    all_AP = []
    all_AP90 = []

    # Validation scores
    if tqdm_pb:
        zip_l = tzip(logits, targets)
    else:
        zip_l = zip(logits, targets)

    id_ = 0
    for logit, target in zip_l:
        print(id_)
        logit = logit.flatten()
        logit_th = np.where(logit > 0.5, 1, 0).astype(np.uint8).flatten()
        target = np.where(target > 0, 1, 0).astype(np.uint8).flatten()

        all_precisions.append(precision_score(target, logit_th))
        all_recall.append(recall_score(target, logit_th))
        all_f1.append(f1_score(target, logit_th))
        print(all_precisions, all_recall, all_f1)

        # Average precision scores
        AP_score, _, _, AP90 = AP(
            logits=[logit], targets=[target], AP90=True, reduce="mean"
        )
        all_AP.append(AP_score)
        all_AP90.append(AP90)
        print(all_AP, all_AP90)
    if reduce == "none":
        return all_precisions, all_recall, all_f1, all_AP, all_AP90
    else:
        assert reduce == "mean"
        return np.mean([all_precisions, all_recall, all_f1, all_AP, all_AP90], axis=1)


def confusion_matrix(logits, targets):
    confusion_vector = logits / targets

    tp = np.sum(confusion_vector == 1)
    fp = np.sum(confusion_vector == float("inf"))
    fn = np.sum(confusion_vector == 0)

    precision = tp / (tp + fp + 1e-16)
    rec = tp / (tp + fn + 1e-16)

    return precision, rec


def call_precision_at_recall(scores: np.ndarray, y: np.ndarray):
    all_th = np.linspace(0, 1, 1000)

    total_p = y.sum()
    scores = np.clip(scores, 0, 1)

    order = np.argsort(scores)
    scores = scores[order]
    y = y[order]

    tp_at_th = np.cumsum(y[::-1])[::-1]
    pp_at_th = np.arange(len(scores), 0, -1)

    indices = np.searchsorted(scores, all_th, side="right")
    # indices[indices == len(tp_at_th)] = len(tp_at_th) - 1
    indices = np.minimum(indices, len(tp_at_th) - 1)

    tp_at_th = tp_at_th[indices]
    pp_at_th = pp_at_th[indices]

    precision = tp_at_th / pp_at_th
    recall = tp_at_th / total_p

    precision[pp_at_th == 0] = 1

    return precision, recall, all_th


def precision_at_recall(precision, recall, all_th, level):
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]
    all_th = all_th[order]

    # mask_1 = recall <= level
    # mask_2 = recall >= level
    #
    # i = np.where(mask_1)[0][-1]
    # j = np.where(mask_2)[0][0]

    # Find the indices of the recall values just below and just above the desired level
    i = np.searchsorted(recall, level, side="right") - 1
    j = np.searchsorted(recall, level, side="left")

    if i < 0:
        i = 0
    if j >= len(recall):
        j = len(recall) - 1

    ri = recall[i]
    rj = recall[j]

    pi = precision[i]
    pj = precision[j]

    ti = all_th[i]
    tj = all_th[j]

    p = pi + (level - ri) / (rj - ri) * (pj - pi)
    th = ti + (level - ri) / (rj - ri) * (tj - ti)

    return p, th


def AP(logits: list, targets: list, AP90=False, reduce="none"):
    assert len(logits) == len(
        targets
    ), "Length of logits and targets should be the same."
    assert reduce in ["none", "mean", "median"]

    all_precisions = np.zeros_like(np.linspace(0, 1, 1000))
    all_recalls = np.zeros_like(np.linspace(0, 1, 1000))
    ap90_scores = []

    for logit, target in zip(logits, targets):
        target = np.where(target > 0, 1, 0).astype(np.uint8).flatten()
        logit = logit.flatten()

        precision, recall, th = call_precision_at_recall(scores=logit, y=target)

        all_precisions += precision
        all_recalls += recall

        if AP90:
            ap90_score, th = precision_at_recall(precision, recall, th, 0.9)
            ap90_scores.append(ap90_score)

    all_precisions = all_precisions / len(logits)
    all_recalls = all_recalls / len(logits)
    if reduce == "none":
        if AP90:
            return all_precisions, all_recalls, th, ap90_scores
        return all_precisions, all_recalls, th
    elif reduce == "mean":
        all_precisions = np.mean(all_precisions)
        all_recalls = np.mean(all_recalls)
        ap90_scores = np.mean(ap90_scores)

        if AP90:
            return all_precisions, all_recalls, th, ap90_scores
        return all_precisions, all_recalls, th
    else:
        all_precisions = np.median(all_precisions)
        all_recalls = np.median(all_recalls)
        ap90_scores = np.median(ap90_scores)

        if AP90:
            return all_precisions, all_recalls, th, ap90_scores
        return all_precisions, all_recalls, th
