from typing import Optional

import torch
from tardis_dev.dist_pytorch.dist import DIST
from tardis_dev.spindletorch.spindletorch import build_network
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

        if checkpoint is None:
            print(f'Searching for weight file for {network}_{subtype}...')

            weights = torch.load(get_weights_aws(network,
                                                 subtype,
                                                 model_type),
                                 map_location=device)
        else:
            weights = torch.load(checkpoint,
                                 map_location=device)

        self.model = self._build_model_from_checkpoint(structure=weights['model_struct_dict'])
        self.model.load_state_dict(weights['model_state_dict'])

        weights = None
        self.network = network

    def _build_model_from_checkpoint(self,
                                     structure: dict):
        if 'dist_type' in structure:
            model = DIST(n_out=structure['n_out'],
                         node_input=structure['node_input'],
                         node_dim=structure['node_dim'],
                         edge_dim=structure['edge_dim'],
                         num_layers=structure['num_layers'],
                         num_heads=structure['num_heads'],
                         coord_embed_sigma=structure['coord_embed_sigma'],
                         dropout_rate=structure['dropout_rate'],
                         structure=structure['structure'],
                         predict=True)

        if 'cnn_type' in structure:
            model = build_network(network_type=structure['cnn_type'],
                                  classification=structure['classification'],
                                  in_channel=structure['in_channel'],
                                  out_channel=structure['out_channel'],
                                  img_size=self.img_size,
                                  dropout=structure['dropout'],
                                  num_conv_layers=structure['num_conv_layers'],
                                  conv_scaler=structure['conv_scaler'],
                                  conv_kernel=structure['conv_kernel'],
                                  conv_padding=structure['conv_padding'],
                                  maxpool_kernel=structure['maxpool_kernel'],
                                  layer_components=structure['layer_components'],
                                  num_group=structure['num_group'],
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
                                     node_features=None,
                                     padding_mask=None)
                else:
                    out = self.model(coords=x.to(self.device),
                                     node_features=y.to(self.device),
                                     padding_mask=None)
            else:
                out = self.model(x.to(self.device))

        return out.cpu().detach().numpy()[0, 0, :]
