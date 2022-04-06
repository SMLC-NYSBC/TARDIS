from typing import Optional

import torch

from tardis.spindletorch.utils.aws import get_weights_aws


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
                 checkpoint: Optional[str] = None,
                 network: Optional[str] = None,
                 subtype: Optional[str] = None,
                 threshold: Optional[float] = None,
                 tqdm=False):
        self.model = model
        if checkpoint is None:
            print('Downloading weight file...')
            
            weights = torch.load(get_weights_aws(network, subtype,
                                                 save_weights=False),
                                 map_location=device)
            model.load_state_dict(weights['model_state_dict'])
        else:
            weights = torch.load(checkpoint,
                                 map_location=device)

            model.load_state_dict(weights['model_state_dict'])

        weights = None
        self.model.eval()

        self.device = device
        self.threshold = threshold
        self.tqdm = tqdm

    def _predict(self,
                 x: torch.Tensor):
        with torch.no_grad():
            out = self.model(x.to(self.device))

            return out.cpu().detach().numpy()[0, 0, :]
