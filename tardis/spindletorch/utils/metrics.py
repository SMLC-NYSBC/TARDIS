from typing import Optional

import numpy as np
import torch


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
    """
    # Calculate confusion matrix, if np.array pass warning for
    # dividing by 0
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


def normalize_image(image: np.ndarray):
    image_min = np.min(image)
    image_max = np.max(image)

    if image_min == 0 and image_max == 1:
        return image

    if image_min == 0:
        image = np.where(image > image_min, 1, 0)
    elif image_max == 0:
        image = np.where(image < image_max, 1, 0)

    return image
