
from typing import Optional

import torch


class Predictor:
    """
    WRAPPER FOR PREDICTION

     Args:
         model: Model with loaded pretrained weights.
         device: Device on which to predict.
         threshold: Threshold parameater for output binary prediction.
         twdm: Build with progressbar
     """

    def __init__(self,
                 model,
                 device: str,
                 threshold: Optional[float] = None,
                 tqdm=False):
        self.model = model
        self.model.eval()

        self.device = device
        self.threshold = threshold
        self.tqdm = tqdm

    def _predict(self,
                 x: torch.Tensor):
        with torch.no_grad():
            out = self.model(x.to(self.device))

            return out.cpu().detach().numpy()[0, :]
