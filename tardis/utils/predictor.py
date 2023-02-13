#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from typing import Optional

import numpy as np
import torch

from tardis.dist_pytorch.dist import build_dist_network
from tardis.spindletorch.spindletorch import build_cnn_network
from tardis.utils.aws import get_weights_aws
from tardis.utils.errors import TardisError
from tardis.utils.logo import print_progress_bar, TardisLogo


class Predictor:
    """
    WRAPPER FOR PREDICTION

     Args:
         device (torch.device): Device on which to predict.
         checkpoint (str, Optional): Local weights files.
         network (str, Optional): Optional network type name.
         subtype (str, Optional): Optional model subtype name.
         model_type (str, Optional): Optional model type name.
         img_size (int, Optional): Optional image patch size.
     """

    def __init__(self,
                 device: torch.device,
                 checkpoint: Optional[str] = None,
                 network: Optional[str] = None,
                 subtype: Optional[str] = None,
                 img_size: Optional[int] = None,
                 model_type: Optional[str] = None,
                 sigma: Optional[float] = None):
        self.device = device
        self.img_size = img_size
        if checkpoint is None and network is None:
            TardisError('139',
                        'tardis/utils/predictor.py',
                        'Missing network weights or network name!')

        if checkpoint is None:
            print(f'Searching for weight file for {network}_{subtype}...')

            weights = torch.load(get_weights_aws(network,
                                                 subtype,
                                                 model_type),
                                 map_location=device)
        elif isinstance(checkpoint, dict):
            weights = checkpoint
        else:
            print('Loading weight file...')
            weights = torch.load(checkpoint, map_location=device)

        # Allow overwriting sigma
        if sigma is not None:
            weights['model_struct_dict']['coord_embed_sigma'] = sigma

        self.model = self._build_model_from_checkpoint(
            structure=weights['model_struct_dict']
        )

        self.model.load_state_dict(weights['model_state_dict'])

        del weights  # Cleanup weight file from memory
        self.network = network

    def _build_model_from_checkpoint(self,
                                     structure: dict):
        """
        Use checkpoint metadata to build compatible network

        Args:
            structure (dict): Metadata dictionary with network setting.

        Returns:
            pytorch model: NN pytorch model.
        """
        if 'dist_type' in structure:
            model = build_dist_network(network_type=structure['dist_type'],
                                       structure=structure,
                                       prediction=True)
        elif 'cnn_type' in structure:
            model = build_cnn_network(network_type=structure['cnn_type'],
                                      structure=structure,
                                      img_size=self.img_size,
                                      prediction=True)
        else:
            model = None

        return model.to(self.device)

    def predict(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        General predictor.

        Args:
            x (torch.Tensor): Main feature used for prediction.
            y (torch.Tensor, None): Optional feature used for prediction.

        Returns:
            np.ndarray: Predicted features.
        """
        with torch.no_grad():
            self.model.eval()

            if self.network == 'dist':
                if y is None:
                    out = self.model(coords=x.to(self.device), node_features=None)
                else:
                    out = self.model(coords=x.to(self.device),
                                     node_features=y.to(self.device))

                out = out.cpu().detach().numpy()[0, 0, :]
                g_len = out.shape[0]
                g_range = range(g_len)

                # Overwrite diagonal with 1
                out[g_range, g_range] = np.eye(g_len, g_len)[g_range, g_range]
                return out
            else:
                out = self.model(x.to(self.device))

                return out.cpu().detach().numpy()[0, 0, :]


class BasicPredictor:
    """
    BASIC MODEL PREDICTOR FOR DIST AND CNN

    Args:
        model (nn.Module): ML model build with nn.Module or nn.sequential.
        structure (dict): Model structure as dictionary.
        device (str): Device for prediction.
        predicting_DataLoader (torch.DataLoader): DataLoader with prediction dataset.
        print_setting (tuple): Model property to display in TARDIS progress bar.
    """

    def __init__(self,
                 model,
                 structure: dict,
                 device: str,
                 print_setting: tuple,
                 predicting_DataLoader,
                 classification=False):
        super(BasicPredictor, self).__init__()

        self.model = model.to(device)
        self.device = device
        self.structure = structure

        if 'cnn_type' in self.structure:
            self.classification = classification
            self.nn_name = self.structure['cnn_type']
        elif 'dist_type' in self.structure:
            self.nn_name = self.structure['dist_type']

            if 'node_input' in structure:
                self.node_input = structure['node_input']

        self.predicting_DataLoader = predicting_DataLoader

        # Set-up progress bar
        self.progress_predict = TardisLogo()
        self.print_setting = print_setting

        self.id = 0
        self.predicting_idx = len(self.predicting_DataLoader)

    def _update_progress_bar(self):
        """
        Update entire Tardis progress bar.
        """
        if self.id % 50 == 0:
            self.progress_predict(title=f'{self.nn_name} Predicting module',
                                  text_1=self.print_setting[0],
                                  text_2=self.print_setting[1],
                                  text_3=self.print_setting[2],
                                  text_4=self.print_setting[3],
                                  text_8=print_progress_bar(self.id, self.predicting_idx))

    def run_predictor(self):
        """
        Main prediction loop.
        """
        # Initialize progress bar.
        self.progress_predict(title=f'{self.nn_name} prediction module.',
                              text_2='Predicted image: 0',
                              text_3=print_progress_bar(0, self.predicting_idx))

        self._update_progress_bar()

        """Training block"""
        self.model.eval()
        self._predict()

    def _predict(self):
        pass
