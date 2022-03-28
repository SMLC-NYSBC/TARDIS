import gc
from os.path import join

import numpy as np
import torch
from tifffile import tifffile


class Predictor:
    """
    Wrapper for prediction

     Args:
         model: Model with loaded pretrained weights.
         device: Device on which to predict.
         output: Output directory for saving the files.
         prediction_DataLoader: DataLoader object with data sets for training.
         threshold: Threshold parameater for output binary prediction.
         denois: If True output images will be denois.
         notebook: If True plot progress bar in Jupyter else console.
     """

    def __init__(self,
                 model,
                 device: str,
                 output: str,
                 prediction_DataLoader,
                 threshold: float,
                 notebook=False):
        self.model = model
        self.device = device
        self.output = output
        self.prediction_DataLoader = prediction_DataLoader
        self.threshold = threshold
        self.notebook = notebook

    def run_prediction(self):
        """Prediction block block"""
        self._predict()

    def _predict(self):

        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        batch_iter = tqdm(enumerate(self.prediction_DataLoader),
                          'Predicting',
                          total=len(self.prediction_DataLoader),
                          leave=False)

        for i, (x, name) in batch_iter:
            self.model.eval()
            with torch.no_grad():
                out = self.model(x.to(self.device))

                if self.threshold is not None:
                    if out.shape[1] > 1:
                        out_df = torch.argmax(out, 1)
                    else:
                        out_df = torch.where(out > self.threshold, 1, 0)

                    for j in range(out_df.shape[0]):
                        out_batch = out_df.cpu().detach().numpy()[j, :]
                        name_batch = name[j]

                        if len(out_batch.shape) != 3:
                            out_batch = out_batch[0, :]

                        tifffile.imwrite(join(self.output, name_batch + '.tif'),
                                         np.array(out_batch, 'int8'))
                else:
                    for j in range(out.shape[0]):
                        out_batch = out.cpu().detach().numpy()[j, :]
                        name_batch = name[j]

                        if len(out_batch.shape) != 3:
                            out_batch = out_batch[0, :]

                        tifffile.imwrite(join(self.output, name_batch + '.tif'),
                                         np.array(out_batch, 'float32'))
        gc.collect()
