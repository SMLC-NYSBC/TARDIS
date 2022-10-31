from typing import Optional

import torch
from tardis_dev.spindletorch.spindletorch import build_cnn_network
from tardis_dev.dist_pytorch.dist import build_dist_network
from tardis_dev.utils.aws import get_weights_aws


class Predictor:
    """
    WRAPPER FOR PREDICTION

     Args:
         device: Device on which to predict.
         checkpoint:
         network:
         subtype:
         model_type:
     """

    def __init__(self,
                 device: str,
                 checkpoint: Optional[str] = None,
                 network: Optional[str] = None,
                 subtype: Optional[str] = None,
                 img_size: Optional[int] = None,
                 model_type: Optional[str] = None):
        self.device = device
        self.img_size = img_size
        assert checkpoint is not None or network is not None

        if checkpoint is None:
            print(f'Searching for weight file for {network}_{subtype}...')

            weights = torch.load(get_weights_aws(network,
                                                 subtype,
                                                 model_type),
                                 map_location=device)
        else:
            print('Loading weight file for...')
            weights = torch.load(checkpoint,
                                 map_location=device)

        self.model = self._build_model_from_checkpoint(structure=weights['model_struct_dict'])
        self.model.load_state_dict(weights['model_state_dict'])

        weights = None
        self.network = network

    def _build_model_from_checkpoint(self,
                                     structure: dict):
        if 'dist_type' in structure:
            model = build_dist_network(network_type=structure['dist_type'],
                                       structure=structure,
                                       prediction=True)
        if 'cnn_type' in structure:
            model = build_cnn_network(network_type=structure['cnn_type'],
                                      structure=structure,
                                      img_size=self.img_size,
                                      prediction=True)

        return model.to(self.device)

    def _predict(self,
                 x: torch.Tensor,
                 y: Optional[torch.Tensor] = None):
        with torch.no_grad():
            self.model.eval()

            if self.network == 'dist':
                if y is None:
                    out = self.model(coords=x.to(self.device),
                                     node_features=None)
                else:
                    out = self.model(coords=x.to(self.device),
                                     node_features=y.to(self.device))
            else:
                out = self.model(x.to(self.device))

        return out.cpu().detach().numpy()[0, 0, :]
