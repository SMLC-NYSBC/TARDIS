from tardis.utils.logo import Tardis_Logo, printProgressBar


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
