#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from typing import Union

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, roc_curve


def compare_dict_metrics(last_best_dict: dict, new_dict: dict) -> bool:
    """
    Compare the average metric values of two dictionaries and return whether the new
    dictionary has a higher average than the last best dictionary. The metric
    comparison is performed by calculating the average of the dictionary values.

    :param last_best_dict: A dictionary representing the last best metrics. Values must
                           be numeric and will be used to compute their average.
    :param new_dict: A dictionary representing the new metrics. Values must be numeric
                     and will be used to compute their average.
    :return: A boolean indicating whether the average value of metrics in the new
             dictionary is greater than the average value of the last best
             dictionary.
    """
    compare_dict_metric = lambda metric_dict: sum(metric_dict.values()) / len(
        metric_dict
    )

    last_best_dict = compare_dict_metric(last_best_dict)
    new_dict = compare_dict_metric(new_dict)

    # Compare the average metric values and return the result
    return new_dict > last_best_dict


def eval_graph_f1(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float, soft=False
):
    """
    Evaluates the Graph-F1 metric for given logits and targets using a threshold-based
    approach with an option to calculate the soft variant.

    The function supports both soft approximation and threshold-based calculation
    to determine precision, recall, accuracy, and F1-score of predictions. It uses
    specific configurations for masking the diagonal elements and applies varied
    thresholds to optimize performance metrics. This is particularly useful in
    evaluating graph-based predictions.

    :param logits: Prediction scores or probabilities from the model (e.g., output of a neural network).
    :param targets: True binary labels corresponding to the predictions.
    :param threshold: A threshold value for converting logits into binary predictions. Determines decision boundaries.
    :param soft: If True, computes a soft approximation of metrics; otherwise, uses threshold-based evaluation.
    :return: If `soft` is True, returns precision cost, recall cost, and F1 cost as tensors.
             Otherwise, returns averaged accuracy score, precision score, recall score, F1 score, and selected threshold.
    """
    """Mask Diagonal as TP"""
    g_len = logits.shape[1]

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

            id_ = int(threshold * 100)
            accuracy_score = round(acc[id_], 2)
            precision_score = round(precision[id_], 2)
            recall_score = round(recall[id_], 2)
            F1_score = round(f1[id_], 2)
        else:
            accuracy_score, precision_score, recall_score, F1_score = 0, 0, 0, 0
            input_df = torch.where(logits >= threshold, 1, 0)
            idx_1 = torch.where(targets == 1)
            idx_0 = torch.where(targets == 0)

            # IDX_1
            input_ = input_df[idx_1]
            target_ = targets[idx_1]

            tp, fp, tn, fn = confusion_matrix(input_, target_)
            tp = tp - g_len

            accuracy_score += (tp + tn) / (tp + tn + fp + fn + 1e-16)
            precision_score += tp / (tp + fp + 1e-16)
            recall_score += tp / (tp + fn + 1e-16)
            F1_score += (
                2
                * (precision_score * recall_score)
                / (precision_score + recall_score + 1e-16)
            )

            # IDX_0
            input_ = input_df[idx_0]
            target_ = targets[idx_0]

            tn, fn, tp, fp = confusion_matrix(input_, target_)

            acc = (tp + tn) / (tp + tn + fp + fn + 1e-16)
            accuracy_score += acc
            prec = tp / (tp + fp + 1e-16)
            precision_score += prec
            rec = tp / (tp + fn + 1e-16)
            recall_score += rec
            F1_score += 2 * (prec * rec) / (prec + rec + 1e-16)

        return (
            accuracy_score / 2,
            precision_score / 2,
            recall_score / 2,
            F1_score / 2,
            threshold,
        )


def calculate_f1(
    logits: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    best_f1=True,
):
    """
    Calculates evaluation metrics for binary classification tasks, such as F1 score,
    precision, recall, and accuracy. Depending on the `best_f1` flag, the function
    either computes metrics for a specific threshold or iteratively finds the
    threshold yielding the best F1 score.

    :param logits: Predictions or model outputs, either as probabilities or logits.
    :type logits: Union[numpy.ndarray, torch.Tensor]
    :param targets: Ground-truth binary targets for the classification task.
    :type targets: Union[numpy.ndarray, torch.Tensor]
    :param best_f1: Flag to enable iterative calculation to find the threshold
        for the highest F1 score. If False, metrics are calculated directly.
    :type best_f1: bool
    :return: A tuple containing metrics - accuracy, precision, recall, F1 score,
        and (if applicable) the best threshold.
    :rtype: Tuple[float, float, float, float, Optional[float]]
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

        id_ = int(threshold * 100)
        accuracy_score = round(acc[id_], 2)
        precision_score = round(precision[id_], 2)
        recall_score = round(recall[id_], 2)
        F1_score = round(f1[id_], 2)

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
        F1_score = 2 * (
            (precision_score * recall_score) / (precision_score + recall_score + 1e-16)
        )

        return accuracy_score, precision_score, recall_score, F1_score


def AP(logits: np.ndarray, targets: np.ndarray) -> float:
    logits = logits.flatten()
    targets = targets.flatten()

    return average_precision_score(targets, logits)


def AP_instance(input_n: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute the average precision (AP) for the given input and target instances. The function
    compares the input predictions with ground truth instances and calculates precision values
    based on the best matches. Precision is evaluated as the ratio of true positives to the total
    number of positive detections. The final AP is normalized by the total number of unique target
    instances.

    :param input_n: A 2D numpy array representing predicted instances, where the first column
        corresponds to instance labels and the remaining columns represent associated features.
    :param targets: A 2D numpy array representing ground truth (GT) instances, where the first
        column corresponds to instance labels and the remaining columns represent associated features.
    :return: The computed average precision (AP) as a float, normalized across all unique GT
        instances in the targets dataset.
    """
    prec = 0

    # Get GT instances, compute IoU for best mache between GT and input
    for j in np.unique(targets[:, 0]):
        true_c = targets[np.where(targets[:, 0] == j)[0], 1:]  # Pick GT instance
        prec_df = []

        # Select max Prec (best mach)
        for i in np.unique(input_n[:, 0]):
            pred = input_n[np.where(input_n[:, 0] == i)[0]]  # Pick input instance

            # Prec is ratio of true positives to the total number of positive detections
            prec_df.append(
                sum([True for i in true_c if i in pred[:, 1:]]) / len(true_c)
            )
        prec += np.max(prec_df)

    return 1 / (len(np.unique(targets[:, 0]) + 1e-16)) * prec


def AUC(logits: np.ndarray, targets: np.ndarray, diagonal=False) -> float:
    """
    Computes the Area Under the Curve (AUC) for given logits and targets.

    This function calculates the AUC score, which is a measure of the performance
    of a classification model. It uses logits and targets as input and computes
    the Receiver Operating Characteristic (ROC) curve for evaluation. If the
    `diagonal` parameter is set to True, it modifies the diagonal of the input
    logits and targets matrices to ensure specific conditions before calculating
    the AUC.

    :param logits: The predicted scores or probabilities as a numpy
        array. Can be 2D or 3D.
    :param targets: Ground truth binary labels as a numpy array. Must
        match the shape of `logits`.
    :param diagonal: A boolean indicating if the diagonal elements of
        logits/targets matrices should be altered before computation.
        Default is `False`.
    :return: The computed AUC metric as a float.
    """
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


def IoU(input_n: np.ndarray, targets: np.ndarray, diagonal=False):
    """
    Compute the Intersection Over Union (IoU) metric for given input and targets.

    The IoU is a performance metric commonly used for evaluating segmentation models
    in computer vision. It measures the overlap between the ground-truth target
    and predicted values. The optional 'diagonal' parameter allows modifications
    by excluding diagonal elements in the computation, particularly useful in
    multi-class datasets.

    :param input_n: Input numpy array, typically model predictions. Can be a 2D or 3D array.
    :param targets: Target numpy array, representing the ground-truth labels. Must match
        the shape of the input.
    :param diagonal: Flag to enforce diagonal elements to be 1. If True, modifies the
        diagonal elements in the input and targets arrays.
    :return: Computed IoU value as a floating-point number.
    """
    if diagonal:
        g_len = input_n.shape[1]
        g_range = range(g_len)

        if input_n.ndim == 3:
            input_n[:, g_range, g_range] = 1.0
            targets[:, g_range, g_range] = 1.0
        else:
            input_n[g_range, g_range] = 1.0
            targets[g_range, g_range] = 1.0

    input_n = input_n.flatten()
    targets = targets.flatten()

    tp, fp, tn, fn = confusion_matrix(input_n, targets)

    return tp / (tp + fp + fn + 1e-16)


def mcov(
    input_n,
    targets,
):
    """
    Calculate the mean Coverage (mCov) and weighted mean Coverage (mwCov) for given input and target data.

    This function computes the coverage metrics evaluating how well given input instances match with
    ground truth (GT) instances based on Intersection over Union (IoU). It considers both the ratio of
    the instance size (w_g) relative to the total size and overall mean matching.


    :param input_n: Input point cloud data where the first column identifies instance labels and the remaining
        columns represent corresponding coordinates or features.
    :type input_n: numpy.ndarray
    :param targets: Ground truth (GT) point cloud data with the first column identifying instance labels
        and the remaining columns representing corresponding coordinates or features.
    :type targets: numpy.ndarray
    :return: A tuple containing:
        - mCov (float): Mean Coverage across all GT instances.
        - mwCov (float): Weighted Mean Coverage based on instance ratio.
    :rtype: tuple[float, float]
    """
    mCov = 0
    mwCov = 0

    unique_target = np.unique(targets[:, 0])
    G = len(unique_target)
    unique_input = np.unique(input_n[:, 0])

    # Get GT instances, compute IoU for best mach between GT and input
    for j in unique_target:
        g = targets[targets[:, 0] == j, 1:]  # Pick GT instance
        w_g = g.shape[0] / targets.shape[0]  # ratio of instance to whole PC
        iou = []

        # Select max IoU (the best mach)
        for i in unique_input:
            p = input_n[input_n[:, 0] == i, 1:]  # Pick input instance

            # Intersection of coordinates between GT and input instances
            # Union of coordinates between GT and input instances

            intersection = np.sum(np.any(np.isin(p, g), axis=1))
            if intersection > 0:
                union = np.unique(np.concatenate((p, g), axis=0), axis=0).shape[0]
                iou.append(intersection / union)

        if len(iou) > 0:
            max_iou = np.max(iou)
        else:
            max_iou = 0.0

        mwCov += w_g * max_iou
        mCov += max_iou

    return mCov / G, mwCov


def confusion_matrix(
    logits: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
):
    """
    Calculates the confusion matrix components for the provided logits and targets.
    The confusion matrix consists of True Positives (TP), False Positives (FP),
    True Negatives (TN), and False Negatives (FN). This function handles both
    PyTorch tensors and NumPy arrays as input for logits and targets.

    :param logits: The predicted values, supporting either PyTorch tensors or
        NumPy arrays.
    :param targets: The ground-truth values, supporting either PyTorch tensors
        or NumPy arrays.
    :return: A tuple containing four integer values representing True Positives
        (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN),
        computed based on the provided logits and targets.
    :rtype: Tuple[int, int, int, int]
    """
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
    Normalizes a given image represented as a NumPy array.

    This function ensures that the input image array falls within the defined
    binary normalization context. If the image has minimum and maximum values
    already set to 0 and 1, respectively, it is returned unchanged. Otherwise,
    based on the minimum and maximum values, the image is normalized to binary
    values (0 or 1).

    :param image: The input image to be normalized, represented as a NumPy array.
    :return: The normalized image array with binary values, where pixel values are
        either 0 or 1.
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
