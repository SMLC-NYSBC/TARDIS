from typing import Optional

import torch
from tardis.utils.aws import get_weights_aws


class Predictor:
    """
    WRAPPER FOR PREDICTION

     Args:
         model: Model with loaded pretrained weights.
         device: Device on which to predict.
         threshold: Threshold parameater for output binary prediction.
         tqdm: Build with progressbar
     """

    def __init__(self,
                 model,
                 device: str,
                 checkpoint: Optional[str] = None,
                 network: Optional[str] = None,
                 subtype: Optional[str] = None,
                 model_type: Optional[str] = None,
                 tqdm=False):
        self.model = model.to(device)

        if checkpoint is None:
            print(f'Downloading weight file for {network}_{subtype}...')

            weights = torch.load(get_weights_aws(network,
                                                 subtype,
                                                 model_type,
                                                 save_weights=False),
                                 map_location=device)
            model.load_state_dict(weights['model_state_dict'])
        else:
            weights = torch.load(checkpoint,
                                 map_location=device)

            model.load_state_dict(weights['model_state_dict'])

        weights = None
        self.model.eval()

        self.network = network
        self.device = device
        self.tqdm = tqdm

    def _predict(self,
                 x: torch.Tensor):
        with torch.no_grad():
            if self.network == 'graphformer':
                out = self.model(coords=x,
                                 node_features=None,
                                 padding_mask=None)
                return out.cpu().detach().numpy()[0, 0, :]
            else:
                out = self.model(x.to(self.device))

                return out.cpu().detach().numpy()[0, 0, :]
