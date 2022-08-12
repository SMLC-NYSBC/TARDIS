from os.path import join

import numpy as np
import torch
from tardis.slcpy.utils.segment_point_cloud import GraphInstance
from tqdm import tqdm


class Predictor:
    """
    MAIN MODULE THAT BUILD PREDICTION FOR GRAPHFORMER

    Args:
        model: Graphformer model with loaded pre-train weights
        node_input: If True node input (image patches) is provided
        device: Device on which the prediction is done
        prediction_DataLoader: DataLoader for prediction dataset
        output: Directory to save output files
        type: Type of save output
    """

    def __init__(self,
                 model,
                 node_input: bool,
                 device: str,
                 prediction_DataLoader,
                 threshold: float,
                 interactions: int,
                 output: str,
                 type: str):
        self.model = model
        self.node_input = node_input
        self.device = device
        self.prediction_DataLoader = prediction_DataLoader
        self.output = output
        self.type = type
        self.segmenter = GraphInstance(threshold=threshold,
                                       max_interactions=interactions)

    def run_predict(self):
        self.model.eval()
        self.predict()

    def predict(self):
        predict_progress = tqdm(enumerate(self.prediction_DataLoader),
                                "Prediction:",
                                ascii=True,
                                leave=True,
                                dynamic_ncols=True)
        for i, (x, y, name) in predict_progress:
            with torch.no_grad():
                coord_pred = []
                graphs_pred = []
                input_coord, input_img = x.to(self.device), y.to(self.device)

                for j, input in enumerate(input_coord):
                    if self.node_input:
                        logit = self.model(coords=input,
                                           node_features=input_img[j],
                                           padding_mask=None)
                    else:
                        logit = self.model(coords=input,
                                           node_features=None,
                                           padding_mask=None)
                    graphs_pred.append(logit[0, 0, :].cpu().detach().numpy())  # [Length x Length]
                    coord_pred.append(input[0, :].cpu().detach().numpy())  # [Length x Dim]

                # [Length x Dim] XY or XYZ
                logits_segmented = self.segmenter.segment_voxals(graph_voxal=graphs_pred,
                                                                 coord_voxal=coord_pred)

                if self.type == 'csv':
                    np.savetxt(join(self.output, name[0] + '_Segmented.csv'),
                               logits_segmented,
                               delimiter=",")
