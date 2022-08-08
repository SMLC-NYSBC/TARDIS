from typing import Optional

import numpy as np
import sklearn.metrics as metric
import torch
from tardis.spindletorch.utils.utils import normalize_image
from scipy.spatial.distance import cdist


def calculate_F1(input: Optional[np.ndarray] = torch.Tensor,
                 target: Optional[np.ndarray] = torch.Tensor,
                 best_f1=True,
                 smooth=1e-16):
    """
    MODULE USED FOR CALCULATING TRAINING METRICS

    Works with torch an numpy dataset.

    Args:
        input: Prediction output from the model
        target: Ground truth mask
        best_f1: if True an expected inputs is probability of classes and
            measured metrics is soft-f1
        smooth: Arbitrate low number to avoid division by 0
    """
    """Find best f1 based on variating threshold"""
    if best_f1:
        threshold = 0
        f1 = []
        acc = []
        precision = []
        recall = []

        while threshold != 1:
            input_df = torch.where(input > threshold, 1, 0)

            confusion_vector = input_df / target

            tp = torch.sum(confusion_vector == 1).item()
            fp = torch.sum(confusion_vector == float('inf')).item()
            tn = torch.sum(torch.isnan(confusion_vector)).item()
            fn = torch.sum(confusion_vector == 0).item()

            accuracy_score = (tp + tn) / (tp + tn + fp + fn + smooth)
            prec = tp / (tp + fp + smooth)
            rec = tp / (tp + fn + smooth)

            acc.append(accuracy_score)
            precision.append(prec)
            recall.append(rec)
            f1.append(2 * (prec * rec) / (prec + rec + smooth))

            threshold = round(threshold + 0.01, 2)

        threshold = np.where(f1 == np.max(f1))[0]
        if threshold.shape[0] > 1:
            th = 0

            for i in threshold:
                th = th + round(i * 0.01, 2)

            threshold = round(th / len(threshold), 2)
        else:
            threshold = round(float(threshold) * 0.01, 2)

        id = int(100 - int(threshold * 100)) - 1
        accuracy_score = round(acc[id], 2)
        precision_score = round(precision[id], 2)
        recall_score = round(recall[id], 2)
        F1_score = round(f1[id], 2)

        return accuracy_score, precision_score, recall_score, F1_score, threshold
    else:
        if torch.is_tensor(input):
            confusion_vector = input / target

            tp = torch.sum(confusion_vector == 1).item()
            fp = torch.sum(confusion_vector == float('inf')).item()
            tn = torch.sum(torch.isnan(confusion_vector)).item()
            fn = torch.sum(confusion_vector == 0).item()
        else:
            input = normalize_image(input)
            target = normalize_image(target)

            with np.errstate(divide='ignore', invalid='ignore'):
                confusion_vector = input / target

            tp = np.sum(confusion_vector == 1)
            fp = np.sum(confusion_vector == float('inf'))
            tn = np.sum(np.isnan(confusion_vector) is True)
            fn = np.sum(confusion_vector == 0)

        """Metric calculation"""
        # Accuracy Score - (tp + tn) / (tp + tn + fp + fn)
        accuracy_score = (tp + tn) / (tp + tn + fp + fn + smooth)

        # Precision Score - tp / (tp + fp)
        precision_score = tp / (tp + fp + smooth)

        # Recall Score - tp / (tp + tn)
        recall_score = tp / (tp + fn + smooth)

        # F1 Score - 2 * [(Prec * Rec) / (Prec + Rec)]
        F1_score = 2 * ((precision_score * recall_score) /
                        (precision_score + recall_score + smooth))

        return accuracy_score, precision_score, recall_score, F1_score


def F1_metric(target: np.ndarray,
              logits: np.ndarray):
    """ Calculate confusion matrix """
    with np.errstate(divide='ignore', invalid='ignore'):
        confusion_vector = logits / target

    tp = np.sum(confusion_vector == 1).item()
    fp = np.sum(confusion_vector == float('inf')).item()
    tn = np.sum(np.isnan(confusion_vector)).item()
    fn = np.sum(confusion_vector == 0).item()

    """Accuracy Score - (tp + tn) / (tp + tn + fp + fn)"""
    accuracy_score = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    """Precision Score - tp / (tp + fp)"""
    precision_score = tp / (tp + fp + 1e-8)

    """Recall Score - tp / (tp + tn)"""
    recall_score = tp / (tp + fn + 1e-8)

    """F1 Score - 2 * [(Prec * Rec) / (Prec + Rec)]"""
    F1_score = 2 * ((precision_score * recall_score) / (precision_score + recall_score + 1e-8))

    return accuracy_score, precision_score, recall_score, F1_score


def mCov(target: np.ndarray,
         logits: np.ndarray):
    iou, prec, rec = [], [], []
    ap50 = []

    for j in np.unique(target[:, 0]):
        true_c = target[np.where(target[:, 0] == j)[0], 1:]
        df = []

        for i in np.unique(logits[:, 0]):
            pred = logits[np.where(logits[:, 0] == i)[0]]

            intersection = np.sum([True for i in true_c if i in pred[:, 1:]])
            union = np.unique(np.vstack((true_c, pred[:, 1:])), axis=0).shape[0]

            df.append(intersection / union)

        ap50.append(np.max(df))
        id = np.where(df == np.max(df))[0]
        pred = logits[np.where(logits[:, 0] == id)[0]][:, 1:]

        tp = np.sum([True for i in true_c if i in pred])
        fn = np.sum([True for i in true_c if i not in pred])
        fp = np.sum([True for i in pred if i not in true_c])

        """Precision Score - tp / (tp + fp)"""
        precision_score = tp / (tp + fp + 1e-8)

        """Recall Score - tp / (tp + tn)"""
        recall_score = tp / (tp + fn + 1e-8)

        """IoU score"""
        iou.append(np.max(df))
        prec.append(precision_score)
        rec.append(recall_score)

    return np.mean(iou), np.mean(prec), np.mean(rec), np.mean(ap50)


def mPrec_Rec(target: np.ndarray,
              logits: np.ndarray):
    """ Calculate confusion matrix """
    with np.errstate(divide='ignore', invalid='ignore'):
        confusion_vector = logits / target

    tp = np.sum(confusion_vector == 1).item()
    fp = np.sum(confusion_vector == float('inf')).item()
    fn = np.sum(confusion_vector == 0).item()

    """Precision Score - tp / (tp + fp)"""
    precision_score = tp / (tp + fp + 1e-8)

    """Recall Score - tp / (tp + tn)"""
    recall_score = tp / (tp + fn + 1e-8)

    return precision_score, recall_score


def IoU(target: np.ndarray,
        logits: np.ndarray):
    miou = []
    for i in np.append(np.arange(0.5, 0.95, 0.05), 0.25):
        logits_df = np.where(logits >= i, 1, 0)

        target = target.flatten()
        logits_df = logits_df.flatten()

        intersection = np.logical_and(target, logits_df)
        union = np.logical_or(target, logits_df)

        iou = np.sum(intersection) / np.sum(union)
        miou.append(iou)

    return np.mean(miou)


def AUC(target: np.ndarray,
        logits: np.ndarray,
        graph=True):
    index = np.triu_indices(target.shape[0], k=1)
    target = target[index]
    logits = logits[index]

    if graph:
        fpr, tpr, _ = metric.roc_curve(target,
                                       logits)

        return metric.auc(fpr, tpr)
    else:
        prec, rec, _ = metric.precision_recall_curve(target,
                                                     logits)

        return metric.auc(rec, prec)


def distAUC(coord: np.ndarray,
            target: np.ndarray):
    index = np.triu_indices(target.shape[0], k=1)
    target = target[index]

    dist = cdist(coord, coord)[index]

    prec, rec, _ = metric.precision_recall_curve(target,
                                                 -dist)

    return metric.auc(rec, prec)


def AP(target: np.ndarray,
       logits: np.ndarray,
       threshold=0.5):
    _ = np.where(logits >= threshold, 1, 0)

    index = np.triu_indices(target.shape[0], k=1)
    target = target[index]
    logits = logits[index]

    intersection = np.logical_and(target, logits)
    union = np.logical_or(target, logits)

    iou = np.sum(intersection) / np.sum(union)

    return iou
