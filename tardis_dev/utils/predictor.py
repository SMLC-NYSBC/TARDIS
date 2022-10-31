from typing import Optional

import torch
from tardis.utils.logo import Tardis_Logo, printProgressBar
from tardis_dev.dist_pytorch.dist import build_dist_network
from tardis_dev.spindletorch.spindletorch import build_cnn_network
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


class BasicPredictor:
    """
    BASIC MODEL PREDICTOR FOR DIST AND CNN

    Args:
        model (nn.Module): ML model build with nn.Module or nn.sequential.
        structure (dict): Model structure as dictionary.
        device (torch.device): Device for prediction.
        predicting_DataLoader (torch.DataLoader): DataLoader with prediction dataset.
        print_setting (tuple): Model property to display in TARDIS progress bar.
        cnn (bool): If True expect CNN model parameters.
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
        self.progress_predict = Tardis_Logo()
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
                                  text_8=printProgressBar(self.id,
                                                          self.predicting_idx))

    def run_predictor(self):
        """
        Main prediction loop.
        """
        # Initialize progress bar.
        self.progress_predict(title=f'{self.nn_name} prediction module.',
                              text_2='Predicted image: 0',
                              text_3=printProgressBar(0, self.predicting_idx))

        self._update_progress_bar()

        """Training block"""
        self.model.eval()
        self._predict()

    def _predict(self):
        pass
