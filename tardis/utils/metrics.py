#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from typing import Optional, Union

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, roc_curve


def compare_dict_metrics(last_best_dict: dict, new_dict: dict) -> bool:
    """
    Compares two metric dictionaries and returns the one with the highest
    average metric values.

    Args:
        last_best_dict (dict): The previous best metric dictionary.
        new_dict (dict): The new metric dictionary to compare.

    Returns:
        bool: True if the new dictionary has a higher average metric value.
    """
    compare_dict_metric = lambda metric_dict: sum(metric_dict.values()) / len(metric_dict)

    last_best_dict = compare_dict_metric(last_best_dict)
    new_dict = compare_dict_metric(new_dict)

    # Compare the average metric values and return the result
    return new_dict > last_best_dict


def eval_graph_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float, soft=False):
    """
    Module used for calculating training metrics

    Works with torch a numpy dataset.

    Args:
        logits (np.ndarray, torch.Tensor): Prediction output from the model.
        targets (np.ndarray, torch.Tensor): Ground truth mask.
        threshold (float):
        soft:
    """
    """Mask Diagonal as TP"""
    g_len = logits.shape[1]
    g_range = range(g_len)

    if logits.ndim == 3:
        logits[:, g_range, g_range] = 1.0
        targets[:, g_range, g_range] = 1.0
    else:
        logits[g_range, g_range] = 1.0
        targets[g_range, g_range] = 1.0

    if soft:
        logits = torch.flatten(logits)
        targets = torch.flatten(targets)

        tp = torch.sum(logits * targets)
        fp = torch.sum(logits * (1 - targets))
        fn = torch.sum((1 - logits) * targets)
        tn = torch.sum((1 - logits) * (1 - targets))

        soft_prec1 = 1 - (tp / (tp + fp + 1e-16))
        soft_prec0 = 1 - (tn / (tn + fp + 1e-16))

        soft_rec1 = 1 - (tp / (tp + fn + 1e-16))
        soft_rec0 = 1 - (tn / (tn + fn + 1e-16))

        soft_f1_class1 = 1 - (2 * tp / (2 * tp + fn + fp + 1e-16))
        soft_f1_class0 = 1 - (2 * tn / (2 * tn + fn + fp + 1e-16))

        prec_cost = torch.mean(0.5 * (soft_prec1 + soft_prec0))
        rec_cost = torch.mean(0.5 * (soft_rec1 + soft_rec0))
        f1_cost = torch.mean(0.5 * (soft_f1_class1 + soft_f1_class0))
        return prec_cost, rec_cost, f1_cost
    else:
        """Find best f1 based on variation threshold"""
        f1 = []
        acc = []
        precision = []
        recall = []

        if threshold == 0.0:
            while threshold != 1:
                input_df = torch.where(logits > threshold, 1, 0)

                tp, fp, tn, fn = confusion_matrix(input_df, targets)
                tp = tp - len(input_df)  # remove diagonal from F1

                accuracy_score = (tp + tn) / (tp + tn + fp + fn + 1e-16)
                prec = tp / (tp + fp + 1e-16)
                rec = tp / (tp + fn + 1e-16)

                acc.append(accuracy_score)
                precision.append(prec)
                recall.append(rec)
                f1.append(2 * (prec * rec) / (prec + rec + 1e-16))

                threshold = round(threshold + 0.01, 2)

            threshold = np.where(f1 == np.max(f1))[0]

            if threshold.shape[0] > 1:
                th = 0

                for i in threshold:
                    th = th + round(i * 0.01, 2)

                threshold = round(th / len(threshold), 2)
            else:
                threshold = round(float(threshold) * 0.01, 2)

            id = int(threshold * 100)
            accuracy_score = round(acc[id], 2)
            precision_score = round(precision[id], 2)
            recall_score = round(recall[id], 2)
            F1_score = round(f1[id], 2)
        else:
            input_df = torch.where(logits > threshold, 1, 0)

            tp, fp, tn, fn = confusion_matrix(input_df, targets)
            tp = tp - len(input_df)  # remove diagonal from F1

            accuracy_score = (tp + tn) / (tp + tn + fp + fn + 1e-16)
            precision_score = tp / (tp + fp + 1e-16)
            recall_score = tp / (tp + fn + 1e-16)
            F1_score = (
                2 * (precision_score * recall_score) / (precision_score + recall_score + 1e-16)
            )

        return accuracy_score, precision_score, recall_score, F1_score, threshold


def calculate_f1(
    logits: Optional[Union[np.ndarray, torch.Tensor]],
    targets: Optional[Union[np.ndarray, torch.Tensor]],
    best_f1=True,
):
    """
    Module used for calculating training metrics

    Works with torch a numpy dataset.

    Args:
        logits (np.ndarray, torch.Tensor): Prediction output from the model.
        targets (np.ndarray, torch.Tensor): Ground truth mask.
        best_f1 (bool): If True an expected inputs is probability of classes and
            measured metrics is soft-f1.
    """
    """Find best f1 based on variation threshold"""
    if best_f1:
        threshold = 0
        f1 = []
        acc = []
        precision = []
        recall = []

        while threshold != 1:
            input_df = torch.where(logits > threshold, 1, 0)

            tp, fp, tn, fn = confusion_matrix(input_df, targets)

            accuracy_score = (tp + tn) / (tp + tn + fp + fn + 1e-16)
            prec = tp / (tp + fp + 1e-16)
            rec = tp / (tp + fn + 1e-16)

            acc.append(accuracy_score)
            precision.append(prec)
            recall.append(rec)
            f1.append(2 * (prec * rec) / (prec + rec + 1e-16))

            threshold = round(threshold + 0.01, 2)

        threshold = np.where(f1 == np.max(f1))[0]
        if threshold.shape[0] > 1:
            th = 0

            for i in threshold:
                th = th + round(i * 0.01, 2)

            threshold = round(th / len(threshold), 2)
        else:
            threshold = round(float(threshold) * 0.01, 2)

        id = int(threshold * 100)
        accuracy_score = round(acc[id], 2)
        precision_score = round(precision[id], 2)
        recall_score = round(recall[id], 2)
        F1_score = round(f1[id], 2)

        return accuracy_score, precision_score, recall_score, F1_score, threshold
    else:
        tp, fp, tn, fn = confusion_matrix(logits, targets)

        """Accuracy Score - (tp + tn) / (tp + tn + fp + fn)"""
        accuracy_score = (tp + tn) / (tp + tn + fp + fn + 1e-16)

        """Precision Score - tp / (tp + fp)"""
        precision_score = tp / (tp + fp + 1e-16)

        """Recall Score - tp / (tp + tn)"""
        recall_score = tp / (tp + fn + 1e-16)

        """F1 Score - 2 * [(Prec * Rec) / (Prec + Rec)]"""
        F1_score = 2 * ((precision_score * recall_score) / (precision_score + recall_score + 1e-16))

        return accuracy_score, precision_score, recall_score, F1_score


def AP(logits: np.ndarray, targets: np.ndarray) -> float:
    logits = logits.flatten()
    targets = targets.flatten()

    return average_precision_score(targets, logits)


def AP_instance(input: np.ndarray, targets: np.ndarray) -> float:
    prec = 0

    # Get GT instances, compute IoU for best mache between GT and input
    for j in np.unique(targets[:, 0]):
        true_c = targets[np.where(targets[:, 0] == j)[0], 1:]  # Pick GT instance
        prec_df = []

        # Select max Prec (best mach)
        for i in np.unique(input[:, 0]):
            pred = input[np.where(input[:, 0] == i)[0]]  # Pick input instance

            # Prec is ratio of true positives to the total number of positive detections
            prec_df.append(sum([True for i in true_c if i in pred[:, 1:]]) / len(true_c))
        prec += np.max(prec_df)

    return 1 / (len(np.unique(targets[:, 0]) + 1e-16)) * prec


def AUC(logits: np.ndarray, targets: np.ndarray, diagonal=False) -> float:
    if diagonal:
        g_len = logits.shape[1]
        g_range = range(g_len)

        if logits.ndim == 3:
            logits[:, g_range, g_range] = 1.0
            targets[:, g_range, g_range] = 1.0
        else:
            logits[g_range, g_range] = 1.0
            targets[g_range, g_range] = 1.0

    logits = logits.flatten()
    targets = targets.flatten()

    fpr, tpr, _ = roc_curve(targets, logits)
    return auc(fpr, tpr)


def IoU(input: np.ndarray, targets: np.ndarray, diagonal=False):
    if diagonal:
        g_len = input.shape[1]
        g_range = range(g_len)

        if input.ndim == 3:
            input[:, g_range, g_range] = 1.0
            targets[:, g_range, g_range] = 1.0
        else:
            input[g_range, g_range] = 1.0
            targets[g_range, g_range] = 1.0

    input = input.flatten()
    targets = targets.flatten()

    tp, fp, tn, fn = confusion_matrix(input, targets)

    return tp / (tp + fp + fn + 1e-16)


def mcov(
    input: Optional[Union[np.ndarray, torch.Tensor]],
    targets: Optional[Union[np.ndarray, torch.Tensor]],
    weight=False,
) -> float:
    mCov = []
    unique_target = np.unique(targets[:, 0])
    unique_input = np.unique(input[:, 0])

    # Return 0.0 on over segmented PC
    if len(unique_input) > len(unique_target) * 2:
        return 0.0

    # Return 0.0 if only none segmented PC
    if len(unique_input) == 1 and len(unique_target) != 1:
        return 0.0

    # Get GT instances, compute IoU for best mache between GT and input
    for j in unique_target:
        true_c = targets[targets[:, 0] == j, 1:]  # Pick GT instance
        df = []
        if weight:
            w = true_c.shape[0] / targets.shape[0]  # ratio of instance to whole PC

        # Select max IoU (the best mach)
        for i in unique_input:
            pred = input[input[:, 0] == i, 1:]  # Pick input instance

            # Intersection of coordinates between GT and input instances
            intersection = np.sum([True for i in true_c if i in pred])

            # Union of coordinates between GT and input instances
            union = len(np.unique(np.vstack((true_c, pred)), axis=0))

            df.append(intersection / union)

        df = np.max(df)
        if df > 1.0:
            if weight:
                mCov.append(w * 1.0)
            else:
                mCov.append(1.0)
        else:
            if weight:
                mCov.append(w * df)
            else:
                mCov.append(df)  # Pick max IoU for GT instance

    return np.mean(mCov)


def mwcov(
    input: Optional[Union[np.ndarray, torch.Tensor]],
    targets: Optional[Union[np.ndarray, torch.Tensor]],
) -> float:
    return mcov(input, targets, weight=True)


def confusion_matrix(
    logits: Optional[Union[np.ndarray, torch.Tensor]],
    targets: Optional[Union[np.ndarray, torch.Tensor]],
):
    if torch.is_tensor(logits):
        confusion_vector = logits / targets

        tp = torch.sum(confusion_vector == 1).item()
        fp = torch.sum(confusion_vector == float("inf")).item()
        tn = torch.sum(torch.isnan(confusion_vector)).item()
        fn = torch.sum(confusion_vector == 0).item()
    else:
        logits = normalize_image(logits)
        targets = normalize_image(targets)

        with np.errstate(divide="ignore", invalid="ignore"):
            confusion_vector = logits / targets

        tp = np.sum(confusion_vector == 1)
        fp = np.sum(confusion_vector == float("inf"))
        tn = np.sum(np.isnan(confusion_vector) is True)
        fn = np.sum(confusion_vector == 0)

    return tp, fp, tn, fn


def normalize_image(image: np.ndarray):
    """
    Simple image data normalizer between 0,1

    Args:
        image (np.ndarray): Image data set.
    """
    image_min = np.min(image)
    image_max = np.max(image)

    if image_min == 0 and image_max == 1:
        return image

    if image_min == 0:
        image = np.where(image > image_min, 1, 0)
    elif image_max == 0:
        image = np.where(image < image_max, 1, 0)

    return image
