from os import mkdir
from os.path import join
from shutil import rmtree

import numpy as np
import torch
from slcpy.main import slcpy_semantic, slcpy_stitch
from spindletorch.unet.trainer import calculate_F1
from tifffile import tifffile
from tqdm import tqdm


class FindBestThreshold:
    """
    CLASS USED FOR CALCULATING BEST THRESHOLD FOR CLASSIFIER

    The searching mechanism is set up to iterate throw set of thresholds
    set by threshold_step parameter. Starting from threshold = 1, loop is searching
    for the best threshold which is determined by F1 score.

    Args:
        model: Network module with turn on sigmoid as a final layer.
        device: Device name on which training is done.
        temp_output: A directory name for temp. storage.
        threshold_step: Steps for threshold searchin.
        prediction_DataLoader: Dataset used for test prediction with threshold.
    """

    def __init__(self,
                 file_name: str,
                 model,
                 device: str,
                 temp_output: str,
                 prediction_DataLoader):
        self.file_name = file_name
        self.model = model
        self.device = device
        self.temp_output = temp_output
        self.prediction_DataLoader = prediction_DataLoader

        self.threshold_range = np.flip(np.append(np.arange(0.0, 1.0, 0.1), 1))
        self.score = []

    def find_threshold(self,
                       metric: str):
        """ For each threshold predict images and calculate F1 score. """
        batch_iter = tqdm(enumerate(self.threshold_range),
                          'Initial searching for Threshold',
                          leave=False)

        self._build_dir()
        self._predict_with_threshold()
        logits, target = self._stitch_image_build_target()
        
        if logits.shape != target.shape:
            z, y, x = target.shape
            logits = logits[0:z, 0:y, 0:x]

        for i, threshold_next_step in batch_iter:
            logits_th = np.where(logits > threshold_next_step, 1, 0).astype('uint8')
            
            self._calculate_metrics3D(logits=logits_th,
                                      target=target,
                                      metric=metric)

            batch_iter.set_description(f'Threshold {threshold_next_step:.2f} &'
                                       f' {metric}: {self.score[len(self.score) - 1]:.2f}')
            if self.early_stop():
                break

        initail_threshold = self._pick_best_threshold()

        self.threshold_range = np.flip(np.append(np.arange(initail_threshold - 0.1,
                                                           initail_threshold + 0.1,
                                                           0.01),
                                                 1))
        self.score = []
        batch_iter = tqdm(enumerate(self.threshold_range),
                          'Fine searching for Threshold',
                          total=len(self.threshold_range),
                          leave=False)

        for i, threshold_next_step in batch_iter:
            logits_th = np.where(
                logits > threshold_next_step, 1, 0).astype('uint8')
            self._calculate_metrics3D(logits=logits_th,
                                      target=target,
                                      metric=metric)

            batch_iter.set_description(f'Threshold {threshold_next_step:.2f} &'
                                       f' {metric}: {self.score[len(self.score) - 1]:.2f}')
            if self.early_stop():
                break

        del logits, target
        self._clean_dir()

        return self._pick_best_threshold()

    def _predict_with_threshold(self):
        pred_iter = tqdm(enumerate(self.prediction_DataLoader),
                         'Predicting images:',
                         total=len(self.prediction_DataLoader),
                         leave=False)
        # Iterate over each image in dataset
        for i, (x, name) in pred_iter:
            self.model.eval()

            with torch.no_grad():
                out = self.model(x.to(self.device))
                # out = torch.where(out > threshold, 1, 0)

                # Save each image in batch
                for j in range(out.shape[0]):
                    # Reshape to D x H x W
                    out_batch = out.cpu().detach().numpy()[j, 0, :]
                    out_batch = np.array(out_batch, dtype='float16')
                    name_df = name[j]

                    tifffile.imwrite(join(self.temp_output, name_df + '.tif'),
                                     np.array(out_batch, 'float16'))

    def _stitch_image_build_target(self):
        stitched_image = slcpy_stitch(dir_path=self.temp_output,
                                      mask=True,
                                      prefix=None,
                                      dtype='float16')
        _, ground_truth = slcpy_semantic(dir_path=self.file_name,
                                         mask=True,
                                         circle_size=250,
                                         trim_mask=False)
        return np.array(stitched_image, 'float16'), np.array(ground_truth, 'uint8')

    @staticmethod
    def _calculate_metrics2D(logits: np.ndarray,
                             target: np.ndarray,
                             metric: str):

        # Check if logits and target are the same, if not trim
        assert logits.shape == target.shape, 'After trimming logist and target ' \
                                             'are still not equal!'

        """ Calculate score and add it to the list """
        if metric == 'accuracy':
            score, _, _, _ = calculate_F1(logits, target, best_f1=False)
        elif metric == 'precision':
            _, score, _, _ = calculate_F1(logits, target, best_f1=False)
        elif metric == 'recall':
            _, _, score, _ = calculate_F1(logits, target, best_f1=False)
        elif metric == 'f1':
            _, _, _, score = calculate_F1(logits, target, best_f1=False)

        return score

    def _calculate_metrics3D(self,
                             logits: np.ndarray,
                             target: np.ndarray,
                             metric: str):
        z, _, _ = target.shape
        score_total = 0

        for i in range(z):
            score_partial = self._calculate_metrics2D(logits=logits[i, :],
                                                      target=target[i, :],
                                                      metric=metric)
            score_total = score_total + score_partial

        final_score = score_total / z  # Calculate mean for all z slices
        self.score.append(final_score)

    def _build_dir(self):
        mkdir(self.temp_output)

    def _clean_dir(self):
        rmtree(self.temp_output)

    def early_stop(self):
        if len(self.score) > 1:
            score_current = self.score[len(self.score) - 1]
            score_past = self.score[len(self.score) - 2]
            return score_current < score_past

        return False

    def _pick_best_threshold(self):
        """ Simply pick highest F1 and associated threshold"""

        return self.threshold_range[self.score.index(max(self.score))]
