#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
from os import getcwd
from os.path import join

import numpy as np
import torch
from torch import nn

from tardis_em.dist_pytorch.utils.segment_point_cloud import PropGreedyGraphCut
from tardis_em.utils.metrics import eval_graph_f1, mcov
from tardis_em.utils.trainer import BasicTrainer


class SparseDistTrainer(BasicTrainer):
    """
    SparseDistTrainer Class

    The SparseDistTrainer class is designed for training models with a focus on
    graph-based data structures and various optimization pipelines. It extends
    the BasicTrainer class and introduces specific functionalities, such as
    graph-based thresholding and metrics tracking. The class includes utilities
    to save metrics, train the model, and validate its performance. It is
    tailored for models requiring graph propagation and metric-driven checkpoints.

    Initialization involves setting up key components and thresholds for
    various graph configurations. Training and validation methods are provided
    to execute these processes on loaded datasets. The class incorporates
    functions to save metrics, display progress updates, and manage checkpoints
    based on improvements in tracked metrics.
    """

    def __init__(self, **kwargs):
        super(SparseDistTrainer, self).__init__(**kwargs)

        self.node_input = self.structure["node_input"]

        self.Graph_gt = PropGreedyGraphCut(threshold=0.5, connection=1000)
        self.Graph0_25 = PropGreedyGraphCut(
            threshold=0.25, connection=self.instance_cov
        )
        self.Graph0_5 = PropGreedyGraphCut(threshold=0.5, connection=self.instance_cov)
        self.Graph0_9 = PropGreedyGraphCut(threshold=0.9, connection=self.instance_cov)

        self.mCov0_25, self.mCov0_5, self.mCov0_9 = [], [], []

    @staticmethod
    def _update_desc(stop_count: int, metric: list) -> str:
        """
        Constructs a formatted string describing metrics for early stopping, F1 score, and coverage metrics.

        This static method generates a detailed string based on the given stop count and a list of metrics.
        Each metric in the list corresponds to a specific evaluation indicator. The method formats these
        metrics to two decimal places for consistent readability.

        :param stop_count: Number of consecutive epochs where early stopping has occurred.
        :param metric: List containing evaluation metrics in the following order:
                       F1 score (min, max), coverage at 0.5 threshold (min, max),
                       and coverage at 0.9 threshold (min, max).
        :return: A formatted description string summarizing the stop count and metrics.
        :rtype: str
        """
        desc = (
            f"Epochs: early_stop: {stop_count}; "
            f"F1: [{metric[0]:.2f}; {metric[1]:.2f}]; "
            f"mCov 0.5: [{metric[2]:.2f}; {metric[3]:.2f}]; "
            f"mCov 0.9: [{metric[4]:.2f}; {metric[5]:.2f}]"
        )
        return desc

    def _update_epoch_desc(self):
        """
        Updates the epoch description based on the current state, including early stopping
        counter and performance metrics. The function utilizes information about F1 scores
        and other metrics (e.g., mCov0_5, mCov0_9) to set a human-readable description
        of training progress. If it is the first epoch, a default description is assigned.
        """
        # For each Epoch load be t model from previous run
        if self.id == 0:
            self.epoch_desc = "Epochs: early_stop: 0; best F1: NaN"
        else:
            self.epoch_desc = self._update_desc(
                self.early_stopping.counter,
                [
                    np.max(self.f1) if len(self.f1) > 0 else 0.0,
                    self.f1[-1:][0] if len(self.f1) > 0 else 0.0,
                    np.max(self.mCov0_5) if len(self.mCov0_5) > 0 else 0.0,
                    self.mCov0_5[-1:][0] if len(self.mCov0_5) > 0 else 0.0,
                    np.max(self.mCov0_9) if len(self.mCov0_9) > 0 else 0.0,
                    self.mCov0_9[-1:][0] if len(self.mCov0_9) > 0 else 0.0,
                ],
            )

    def _save_metric(self) -> bool:
        """
        Saves metrics and model weights, including training and evaluation metrics,
        model structure, model state, optimizer state, and learning rate. It creates
        CSV files to log training losses, validation losses, evaluation metrics,
        and learning rates in a checkpoint directory. Additionally, it saves model
        checkpoints conditionally based on evaluation criteria such as F1 score and
        mCov score.

        :returns:
            A boolean indicating if early stopping is triggered.
        :rtype: bool

        :raises:
            FileNotFoundError: If specified directories or paths do not exist while
            saving files.
            RuntimeError: If any issue occurs during saving the model or metrics.
        """
        if len(self.training_loss) > 0:
            np.savetxt(
                join(
                    getcwd(),
                    f"{self.checkpoint_name}_checkpoint",
                    "training_losses.csv",
                ),
                self.training_loss,
                delimiter=",",
            )
        if len(self.validation_loss) > 0:
            np.savetxt(
                join(
                    getcwd(),
                    f"{self.checkpoint_name}_checkpoint",
                    "validation_losses.csv",
                ),
                self.validation_loss,
                delimiter=",",
            )
        if len(self.f1) > 0:
            np.savetxt(
                join(getcwd(), f"{self.checkpoint_name}_checkpoint", "eval_metric.csv"),
                np.column_stack(
                    [
                        self.accuracy,
                        self.precision,
                        self.recall,
                        self.threshold,
                        self.f1,
                        self.mCov0_25,
                        self.mCov0_5,
                        self.mCov0_9,
                    ]
                ),
                delimiter=",",
            )
        if len(self.learning_rate) > 0:
            np.savetxt(
                join(
                    getcwd(), f"{self.checkpoint_name}_checkpoint", "learning_rate.csv"
                ),
                self.learning_rate,
                delimiter=",",
            )

        """ Save current model weights"""
        # If mean evaluation f1 score is higher than save checkpoint
        if all(self.f1[-1:][0] >= i for i in self.f1[:-1]):
            torch.save(
                {
                    "model_struct_dict": self.structure,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                join(
                    getcwd(),
                    f"{self.checkpoint_name}_checkpoint",
                    f"{self.checkpoint_name}_checkpoint_f1.pth",
                ),
            )

        # If mean evaluation mcov score is higher than save checkpoint
        if all(self.mCov0_9[-1:][0] >= i for i in self.mCov0_9[:-1]):
            torch.save(
                {
                    "model_struct_dict": self.structure,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                join(
                    getcwd(),
                    f"{self.checkpoint_name}_checkpoint",
                    f"{self.checkpoint_name}_checkpoint_mcov.pth",
                ),
            )

        torch.save(
            {
                "model_struct_dict": self.structure,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            join(getcwd(), f"{self.checkpoint_name}_checkpoint", "model_weights.pth"),
        )

        if self.early_stopping.early_stop:
            return True
        return False

    def _train(self):
        """
        Trains the DIST model using the specified training DataLoader, optimizer, and criterion.
        Includes progress bar updates, mid-training evaluation, and tracking of training metrics
        such as loss and learning rate. For each batch, the model updates weights through backpropagation
        and optionally adjusts learning rates using the provided scheduler.

        :return: None
        """
        # Update progress bar
        self._update_progress_bar(
            loss_desc="Training: (loss 1.000)", idx=0, task="Start Training..."
        )

        # Run training for DIST model
        for idx, (e, n, g, _, _) in enumerate(self.training_DataLoader):
            """Mid-training eval"""
            # self._update_progress_bar(
            #     loss_desc="Training: (loss 1.000)", idx=0, task="Mid-train Eval..."
            # )
            if len(self.training_DataLoader) > 100:
                self._mid_training_eval(idx=idx)

            """Training"""
            for edge, node, graph in zip(e, n, g):
                edge, graph = edge.to(self.device), graph.to(self.device)
                self.optimizer.zero_grad()

                edge, indices = self.model(coord=edge[0, :])
                graph = graph[0, indices[3][:, 0], indices[3][:, 1]].type(torch.float32)

                # Back-propagate
                loss = self.criterion(edge[:, 0], graph)
                loss.backward()  # One backward pass
                self.optimizer.step()  # Update the parameters

                # Store training loss metric
                loss_value = loss.item()
                self.training_loss.append(loss_value)

                # Store and update learning rate
                if self.lr_scheduler:
                    self.lr = self.optimizer.get_lr_scale()
                self.learning_rate.append(self.lr)

                # Update progress bar
                self._update_progress_bar(
                    loss_desc=f"Training: (loss {loss_value:.4f};"
                    f" LR: {self.lr:.5f})",
                    idx=idx,
                    task="Training...",
                )

    def _validate(self):
        """
        _validates the performance and metrics for the validation dataset.

        This private method performs several computations on the validation dataset
        to evaluate the model's performance. It computes validation losses, accuracy,
        precision, recall, F1-score, and additional custom metrics, such as mean
        coverage (mCov) at various thresholds. The method iterates through the
        validation dataset, conducts predictions, calculates metrics, updates the
        progress bar, computes aggregation statistics, and triggers early stopping
        evaluation.

        :return: None
        :rtype: None
        """
        valid_losses = []
        accuracy_mean = []
        precision_mean = []
        recall_mean = []
        F1_mean = []
        threshold_mean = []
        mcov0_25, mcov0_5, mcov0_9 = [], [], []

        for idx, (e, n, g, o, _) in enumerate(self.validation_DataLoader):
            coord = [x.cpu().detach().numpy()[0, :] for x in e]
            edge_cpu, graph_cpu, out_cpu = [], [], []

            for edge, node, graph, out in zip(e, n, g, o):
                edge, graph = edge[0, :].to(self.device), graph.to(self.device)

                with torch.no_grad():
                    out_cpu.append(out.cpu().detach().numpy()[0, :])

                    # Predict graph
                    edge, indices = self.model(coord=edge)

                    # Calculate late validation loss
                    loss = self.criterion(
                        edge[:, 0],
                        graph[0, indices[3][:, 0], indices[3][:, 1]].type(
                            torch.float32
                        ),
                    )

                    # Calculate F1 metric
                    pred_edge = np.zeros(indices[2])
                    pred_edge[indices[3][:, 0], indices[3][:, 1]] += (
                        edge.cpu().detach().numpy()[:, 0]
                    )
                    np.fill_diagonal(pred_edge, 1)
                    edge_cpu.append(pred_edge)
                    graph_cpu.append(graph[0, :].cpu().detach().numpy())

                    acc, prec, recall, f1, th = eval_graph_f1(
                        logits=torch.from_numpy(pred_edge),
                        targets=graph[0, ...].cpu().detach(),
                        threshold=0.5,
                    )

                # Avg. precision score
                valid_losses.append(loss.item())
                accuracy_mean.append(acc)
                precision_mean.append(prec)
                recall_mean.append(recall)
                F1_mean.append(f1)
                threshold_mean.append(th)

                # Update progress bar
                df_05, df_09 = 0.0, 0.0
                if len(mcov0_5) > 0:
                    df_05 = mcov0_5[-1:][0]
                if len(mcov0_9) > 0:
                    df_09 = mcov0_9[-1:][0]

                valid = (
                    f"Validation: (loss: {loss.item():.4f}; F1: {f1:.2f}) "
                    f"mCov[0.5]: {df_05:.2f}; "
                    f"mCov[0.9]: {df_09:.2f}"
                )
                self._update_progress_bar(loss_desc=valid, idx=idx, train=False)

            # Build GT instance point cloud
            target = self.Graph_gt.patch_to_segment(
                graph=graph_cpu, coord=coord, idx=out_cpu, prune=0, sort=False
            )

            # Threshold 0.25
            try:
                input0_1 = self.Graph0_25.patch_to_segment(
                    graph=edge_cpu, coord=coord, idx=out_cpu, prune=5, sort=False
                )
                mcov_m, _ = mcov(input0_1, target)
                mcov0_25.append(mcov_m)
            except:
                mcov0_25.append(0.0)

            # Threshold 0.5
            try:
                input0_5 = self.Graph0_5.patch_to_segment(
                    graph=edge_cpu, coord=coord, idx=out_cpu, prune=5, sort=False
                )
                mcov_m, _ = mcov(input0_5, target)
                mcov0_5.append(mcov_m)
            except:
                mcov0_5.append(0.0)

            # Threshold 0.9
            try:
                input0_9 = self.Graph0_9.patch_to_segment(
                    graph=edge_cpu, coord=coord, idx=out_cpu, prune=5, sort=False
                )
                mcov_m, _ = mcov(input0_9, target)
                mcov0_9.append(mcov_m)
            except:
                mcov0_9.append(0.0)

            # Update progress bar
            valid = (
                f"Validation: (loss: {loss.item():.4f}; F1: {f1:.2f}) "
                f"mCov[0.5]: {mcov0_5[-1:][0]:.2f}; "
                f"mCov[0.9]: {mcov0_9[-1:][0]:.2f}"
            )
            self._update_progress_bar(loss_desc=valid, idx=idx, train=False)

        # Reduce eval. metric with mean
        self.validation_loss.append(np.mean(valid_losses))
        self.accuracy.append(np.mean(accuracy_mean))
        self.precision.append(np.mean(precision_mean))
        self.recall.append(np.mean(recall_mean))
        self.threshold.append(np.mean(threshold_mean))
        self.f1.append(np.mean(F1_mean))

        self.mCov0_25.append(np.mean(mcov0_25))
        self.mCov0_5.append(np.mean(mcov0_5))
        self.mCov0_9.append(np.mean(mcov0_9))

        # Check if average evaluation loss dropped
        self.early_stopping(f1_score=self.f1[-1:][0])


class DistTrainer(BasicTrainer):
    """
    Handles the training process for a distributed graph-based model leveraging
    different segmentation thresholds. This class is designed to optimize the
    training and checkpointing process for models that utilize graph-based
    representations.

    The class uses various greedy graph cut algorithms with different thresholds
    to generate segmentations. Metrics and model states are saved during training
    to ensure progress tracking and checkpointing for better model reproducibility.
    """

    def __init__(self, **kwargs):
        super(DistTrainer, self).__init__(**kwargs)

        self.node_input = self.structure["node_input"]

        self.Graph_gt = PropGreedyGraphCut(threshold=0.5, connection=1000)
        self.Graph0_25 = PropGreedyGraphCut(
            threshold=0.25, connection=self.instance_cov
        )
        self.Graph0_5 = PropGreedyGraphCut(threshold=0.5, connection=self.instance_cov)
        self.Graph0_9 = PropGreedyGraphCut(threshold=0.9, connection=self.instance_cov)

        self.mCov0_25, self.mCov0_5, self.mCov0_9 = [], [], []

    @staticmethod
    def _update_desc(stop_count: int, metric: list) -> str:
        """
        Constructs a descriptive string summarizing the early stopping count and a set of
        performance metrics. The description includes specific metrics formatted for easier
        interpretation of the results, such as F1 scores and mCoverage at thresholds 0.5 and
        0.9, respectively.

        :param stop_count: Early stopping count indicating the number of epochs without
            improvement that triggers early stopping in a training loop.
        :type stop_count: int
        :param metric: A list containing performance metrics formatted to be included in the
            description string. The first two elements represent F1 scores, while the next
            four represent mCoverage metrics at thresholds 0.5 and 0.9, respectively.
        :type metric: list

        :return: A formatted string summarizing the provided early stopping count and
            performance metrics, designed for output or logging purposes.
        :rtype: str

        """
        desc = (
            f"Epochs: early_stop: {stop_count}; "
            f"F1: [{metric[0]:.2f}; {metric[1]:.2f}]; "
            f"mCov 0.5: [{metric[2]:.2f}; {metric[3]:.2f}]; "
            f"mCov 0.9: [{metric[4]:.2f}; {metric[5]:.2f}]"
        )
        return desc

    def _update_epoch_desc(self):
        """
        Updates the epoch description with relevant metrics and stopping information.

        This method updates the `epoch_desc` attribute for the current epoch, calculating
        and incorporating various metrics (e.g., F1 scores, mCov0_5, mCov0_9) and the state
        of early stopping. For the first epoch (when `self.id == 0`), it sets a default
        description. For subsequent epochs, it computes the metrics and formats the
        description using `_update_desc`.

        :param self: Instance of the class containing attributes `id`, `early_stopping`,
            `f1`, `mCov0_5`, and `mCov0_9` required for building `epoch_desc`.
        :type self: Class instance
        """
        # For each Epoch load be t model from previous run
        if self.id == 0:
            self.epoch_desc = "Epochs: early_stop: 0; best F1: NaN"
        else:
            self.epoch_desc = self._update_desc(
                self.early_stopping.counter,
                [
                    np.max(self.f1) if len(self.f1) > 0 else 0.0,
                    self.f1[-1:][0] if len(self.f1) > 0 else 0.0,
                    np.max(self.mCov0_5) if len(self.mCov0_5) > 0 else 0.0,
                    self.mCov0_5[-1:][0] if len(self.mCov0_5) > 0 else 0.0,
                    np.max(self.mCov0_9) if len(self.mCov0_9) > 0 else 0.0,
                    self.mCov0_9[-1:][0] if len(self.mCov0_9) > 0 else 0.0,
                ],
            )

    def _save_metric(self) -> bool:
        """
        Saves various training metrics, evaluation metrics, learning rates, and
        model weights to files, and updates checkpoint files based on evaluation
        criteria. The method validates conditions to save different checkpoints and
        the final model weights. It also determines if training should be stopped
        early.

        :param self: The instance of the class that contains the metrics, learning
            rate, and other checkpoint parameters.

        :raises OSError: Raised if the file save operation fails.
        :raises AttributeError: Raised if required attributes are missing in the
            input object.

        :return: A boolean indicating whether early stopping conditions are met.
        :rtype: bool
        """
        if len(self.training_loss) > 0:
            np.savetxt(
                join(
                    getcwd(),
                    f"{self.checkpoint_name}_checkpoint",
                    "training_losses.csv",
                ),
                self.training_loss,
                delimiter=",",
            )
        if len(self.validation_loss) > 0:
            np.savetxt(
                join(
                    getcwd(),
                    f"{self.checkpoint_name}_checkpoint",
                    "validation_losses.csv",
                ),
                self.validation_loss,
                delimiter=",",
            )
        if len(self.f1) > 0:
            np.savetxt(
                join(getcwd(), f"{self.checkpoint_name}_checkpoint", "eval_metric.csv"),
                np.column_stack(
                    [
                        self.accuracy,
                        self.precision,
                        self.recall,
                        self.threshold,
                        self.f1,
                        self.mCov0_25,
                        self.mCov0_5,
                        self.mCov0_9,
                    ]
                ),
                delimiter=",",
            )
        if len(self.learning_rate) > 0:
            np.savetxt(
                join(
                    getcwd(), f"{self.checkpoint_name}_checkpoint", "learning_rate.csv"
                ),
                self.learning_rate,
                delimiter=",",
            )

        """ Save current model weights"""
        # If mean evaluation f1 score is higher than save checkpoint
        if all(self.f1[-1:][0] >= i for i in self.f1[:-1]):
            torch.save(
                {
                    "model_struct_dict": self.structure,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                join(
                    getcwd(),
                    f"{self.checkpoint_name}_checkpoint",
                    f"{self.checkpoint_name}_checkpoint_f1.pth",
                ),
            )

        # If mean evaluation mcov score is higher than save checkpoint
        if all(self.mCov0_9[-1:][0] >= i for i in self.mCov0_9[:-1]):
            torch.save(
                {
                    "model_struct_dict": self.structure,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                join(
                    getcwd(),
                    f"{self.checkpoint_name}_checkpoint",
                    f"{self.checkpoint_name}_checkpoint_mcov.pth",
                ),
            )

        torch.save(
            {
                "model_struct_dict": self.structure,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            join(getcwd(), f"{self.checkpoint_name}_checkpoint", "model_weights.pth"),
        )

        if self.early_stopping.early_stop:
            return True
        return False

    def _train(self):
        """
        Trains the model using the provided DataLoader and updates metrics such as
        training loss and learning rate. The training process supports mid-training
        evaluations, dynamic learning rate adjustments, and progress bar updates
        for better monitoring.

        Raises errors if required components or methods like the DataLoader, device,
        optimizer, criterion, model, or progress bar are not properly configured or
        defined.

        :return: None
        """
        # Update progress bar
        self._update_progress_bar(
            loss_desc="Training: (loss 1.000)", idx=0, task="Start Training..."
        )

        # Run training for DIST model
        for idx, (e, n, g, _, _) in enumerate(self.training_DataLoader):
            """Mid-training eval"""
            # self._update_progress_bar(
            #     loss_desc="Training: (loss 1.000)", idx=0, task="Mid-train Eval..."
            # )
            if len(self.training_DataLoader) > 100:
                self._mid_training_eval(idx=idx)

            """Training"""
            for edge, node, graph in zip(e, n, g):
                edge, graph = edge.to(self.device), graph.to(self.device)
                self.optimizer.zero_grad()

                if self.node_input > 0:
                    edge = self.model(coords=edge, node_features=node.to(self.device))
                else:
                    edge = self.model(coords=edge, node_features=None)

                # Back-propagate
                loss = self.criterion(edge[:, 0, ...], graph)  # Calc. loss
                loss.backward()  # One backward pass
                self.optimizer.step()  # Update the parameters

                # Store training loss metric
                loss_value = loss.item()
                self.training_loss.append(loss_value)

                # Store and update learning rate
                if self.lr_scheduler:
                    self.lr = self.optimizer.get_lr_scale()
                self.learning_rate.append(self.lr)

                # Update progress bar
                self._update_progress_bar(
                    loss_desc=f"Training: (loss {loss_value:.4f};"
                    f" LR: {self.lr:.5f})",
                    idx=idx,
                    task="Training...",
                )

    @staticmethod
    def _greedy_segmenter(graph: np.ndarray, coord: np.ndarray, th: float):
        """
        Segments a graph into distinct components based on the provided threshold value.
        This method employs a greedy approach to cluster nodes into segments
        by iteratively grouping connected nodes that exceed the defined
        threshold. Each node belongs to one and only one segment, and the
        resulting segments are returned with their corresponding coordinates.

        :param graph: 2D numpy array representing the adjacency matrix of the
            graph. Each value in the matrix indicates the weight of the edge
            between nodes, with zero indicating no connection.
        :param coord: 2D numpy array containing the coordinates of the graph nodes.
            Each row corresponds to a node, and columns store the spatial
            information (e.g., [x, y]).
        :param th: A float representing the threshold value. Only edges with
            weights greater than or equal to this value are considered for
            segmenting the graph.
        :return: A 2D numpy array where each row corresponds to a node. The first
            column contains the segment identifiers, and the remaining columns
            contain the coordinates of the corresponding nodes.
        """
        node = dict()
        for j in range(len(graph)):
            df = [[id_, i] for id_, i in enumerate(graph[j]) if i >= th and id != j]
            node[j] = dict(df)
        coord_list = dict()
        seg_id = 0
        full_list = list(range(len(graph)))

        while len(full_list) != 0:
            id_i, old_id = [full_list[0]], [full_list[0]]
            added = list(node[id_i[0]].keys())
            id_i = id_i + added

            while len(id_i) != len(old_id):
                old_id = id_i
                added_new = []
                for key in added:
                    added_new = added_new + list(node[key].keys())
                added = list(dict.fromkeys(added_new))

                id_i = list(dict.fromkeys(id_i + added))
            id_i = sorted(id_i)

            coord_list[seg_id] = id_i
            seg_id += 1

            for i in id_i:
                full_list.remove(i)

        list_c = []
        for key in coord_list:
            value = coord[coord_list[key], :]
            id_i = np.repeat(key, len(value)).reshape(-1, 1)
            list_c.append(np.hstack((id_i, value)))
        return np.concatenate(list_c)

    def _validate(self):
        """
        Validates the model using the provided validation DataLoader. Measures and calculates several
        metrics during the validation process, including validation loss, accuracy, precision, recall,
        F1 score, thresholds, and mCov metrics. Updates progress bars during validation and performs
        early stopping checks based on the latest F1 score.

        :raises Exception: If an error occurs while calculating mCov values for thresholds 0.25, 0.5,
                           or 0.9. This is handled with a fallback value of 0.0.

        :param self: The class instance containing the necessary data and attributes for validation.

        :return: Updates the class instance with the validation metrics calculated for the current epoch.
        """
        valid_losses = []
        accuracy_mean = []
        precision_mean = []
        recall_mean = []
        F1_mean = []
        threshold_mean = []
        mcov0_25, mcov0_5, mcov0_9 = [], [], []

        for idx, (e, n, g, o, _) in enumerate(self.validation_DataLoader):
            coord = [x.cpu().detach().numpy()[0, :] for x in e]
            edge_cpu, graph_cpu, out_cpu = [], [], []

            for edge, node, graph, out in zip(e, n, g, o):
                edge, graph = edge.to(self.device), graph.to(self.device)

                with torch.no_grad():
                    out_cpu.append(out.cpu().detach().numpy()[0, :])
                    graph_cpu.append(graph[0, :].cpu().detach().numpy())

                    # Predict graph
                    if self.node_input > 0:
                        edge = self.model(
                            coords=edge, node_features=node.to(self.device)
                        )
                    else:
                        edge = self.model(coords=edge, node_features=None)

                    # Calculate validation loss
                    loss = self.criterion(edge[0, :], graph)

                    # Calculate F1 metric
                    edge[0, 0, :].fill_diagonal_(1)
                    acc, prec, recall, f1, th = eval_graph_f1(
                        logits=edge[0, 0, :], targets=graph[0, :], threshold=0.5
                    )
                    edge_cpu.append(edge[0, 0, :].cpu().detach().numpy())

                # Avg. precision score
                valid_losses.append(loss.item())
                accuracy_mean.append(acc)
                precision_mean.append(prec)
                recall_mean.append(recall)
                F1_mean.append(f1)
                threshold_mean.append(th)

                # Update progress bar
                df_05, df_09 = 0.0, 0.0
                if len(mcov0_5) > 0:
                    df_05 = mcov0_5[-1:][0]
                if len(mcov0_9) > 0:
                    df_09 = mcov0_9[-1:][0]

                valid = (
                    f"Validation: (loss: {loss.item():.4f}; F1: {f1:.2f}) "
                    f"mCov[0.5]: {df_05:.2f}; "
                    f"mCov[0.9]: {df_09:.2f}"
                )
                self._update_progress_bar(loss_desc=valid, idx=idx, train=False)

            # Build GT instance point cloud
            target = self.Graph_gt.patch_to_segment(
                graph=graph_cpu, coord=coord, idx=out_cpu, prune=0, sort=False
            )

            # Threshold 0.25
            try:
                input0_1 = self.Graph0_25.patch_to_segment(
                    graph=edge_cpu, coord=coord, idx=out_cpu, prune=5, sort=False
                )
                mcov_m, _ = mcov(input0_1, target)
                mcov0_25.append(mcov_m)
            except:
                mcov0_25.append(0.0)

            # Threshold 0.5
            try:
                input0_5 = self.Graph0_5.patch_to_segment(
                    graph=edge_cpu, coord=coord, idx=out_cpu, prune=5, sort=False
                )
                mcov_m, _ = mcov(input0_5, target)
                mcov0_5.append(mcov_m)
            except:
                mcov0_5.append(0.0)

            # Threshold 0.9
            try:
                input0_9 = self.Graph0_9.patch_to_segment(
                    graph=edge_cpu, coord=coord, idx=out_cpu, prune=5, sort=False
                )
                mcov_m, _ = mcov(input0_9, target)
                mcov0_9.append(mcov_m)
            except:
                mcov0_9.append(0.0)

            # Update progress bar
            valid = (
                f"Validation: (loss: {loss.item():.4f}; F1: {f1:.2f}) "
                f"mCov[0.5]: {mcov0_5[-1:][0]:.2f}; "
                f"mCov[0.9]: {mcov0_9[-1:][0]:.2f}"
            )
            self._update_progress_bar(loss_desc=valid, idx=idx, train=False)

        # Reduce eval. metric with mean
        self.validation_loss.append(np.mean(valid_losses))
        self.accuracy.append(np.mean(accuracy_mean))
        self.precision.append(np.mean(precision_mean))
        self.recall.append(np.mean(recall_mean))
        self.threshold.append(np.mean(threshold_mean))
        self.f1.append(np.mean(F1_mean))

        self.mCov0_25.append(np.mean(mcov0_25))
        self.mCov0_5.append(np.mean(mcov0_5))
        self.mCov0_9.append(np.mean(mcov0_9))

        # Check if average evaluation loss dropped
        self.early_stopping(f1_score=self.f1[-1:][0])


class CDistTrainer(BasicTrainer):
    """
    Trainer class for implementing a custom distance-based training strategy.

    The CDistTrainer class extends the BasicTrainer and provides
    functionality for training and validating a model with a specific
    distance-based loss criterion and evaluation metrics. This trainer is
    designed for node and edge-based input, handling mid-training evaluations,
    early stopping based on F1-score, and updating performance metrics such
    as loss, accuracy, precision, recall, F1 score, and threshold. It supports
    both training and validation phases using DataLoader objects for respective
    datasets.
    """

    def __init__(self, **kwargs):
        super(CDistTrainer, self).__init__(**kwargs)

        if self.structure["dist_type"] == "semantic":
            self.criterion_cls = nn.CrossEntropyLoss()

    def _train(self):
        """
        Trains the model using the data provided by the `training_DataLoader`. This method
        performs forward passes of the model, computes losses using provided criterion
        functions, and back-propagates those losses to update model parameters. The training
        progress and losses are displayed using a progress bar.

        Detailed operations such as mid-training evaluation and progress bar updates are
        handled within this method.

        :param self: The instance of the class calling this method.
        """
        # Update progress bar
        self._update_progress_bar(loss_desc="Training: (loss 1.000)", idx=0)

        for idx, (e, n, g, _, c) in enumerate(self.training_DataLoader):
            """Mid-training eval"""
            self._mid_training_eval(idx=idx)

            for edge, node, cls, graph in zip(e, n, c, g):
                edge, graph = edge.to(self.device), graph.to(self.device)
                cls = cls.to(self.device)
                self.optimizer.zero_grad()

                if self.node_input:
                    node = node.to(self.device)
                    edge, out_cls = self.model(coords=edge, node_features=node)
                else:
                    edge, out_cls = self.model(coords=edge, node_features=None)

                # Back-propagate
                loss = self.criterion(edge[0, :], graph) + self.criterion_cls(
                    out_cls, cls
                )
                loss.backward()  # One backward pass
                self.optimizer.step()  # Update the parameters

                # Store evaluation loss metric
                loss_value = loss.item()
                self.training_loss.append(loss_value)

                # Update progress bar
                self._update_progress_bar(
                    loss_desc=f"Training: (loss {loss_value:.4f})", idx=idx
                )

    def _validate(self):
        """
        Validates the performance of the model on the provided validation dataset. This
        method computes the validation loss, accuracy, precision, recall, F1-measure, and
        threshold for the model using the given evaluation metrics. It updates the progress
        bar during the computation and tracks the results over multiple batches. The early
        stopping mechanism is also utilized based on the average F1 score.

        :raises ValueError: If `self.validation_DataLoader` is not properly loaded.
        :raises RuntimeError: If the model or the tensors are not properly transferred to the
            specified device before evaluation.
        :raises AttributeError: If `criterion`, `criterion_cls`, or other attributes are undefined.
        """
        valid_losses = []
        accuracy_mean = []
        precision_mean = []
        recall_mean = []
        F1_mean = []
        threshold_mean = []

        for idx, (e, n, g, _, c) in enumerate(self.validation_DataLoader):
            for edge, node, cls, graph in zip(e, n, c, g):
                edge, graph = edge.to(self.device), graph.to(self.device)
                cls = cls.to(self.device)

                with torch.no_grad():
                    if self.node_input:
                        node = node.to(self.device)
                        edge, out_cls = self.model(coords=edge, node_features=node)
                    else:
                        edge, out_cls = self.model(coords=edge, node_features=None)

                    loss = self.criterion(edge[0, :], graph) + self.criterion_cls(
                        out_cls, cls
                    )

                    edge = torch.sigmoid(edge[:, 0, :])
                    acc, prec, recall, f1, th = eval_graph_f1(
                        logits=edge, targets=graph, threshold=0.5
                    )

                # Avg. precision score
                valid_losses.append(loss.item())
                accuracy_mean.append(acc)
                precision_mean.append(prec)
                recall_mean.append(recall)
                F1_mean.append(f1)
                threshold_mean.append(th)
                valid = (
                    "Validation: "
                    f"(loss {loss.item():.4f} Prec: {prec:.2f} Rec: {recall:.2f} F1: {f1:.2f})"
                )

                # Update progress bar
                self._update_progress_bar(
                    loss_desc=valid, idx=idx, task="Validation..."
                )
        # Reduce eval. metric with mean
        self.validation_loss.append(np.mean(valid_losses))
        self.accuracy.append(np.mean(accuracy_mean))
        self.precision.append(np.mean(precision_mean))
        self.recall.append(np.mean(recall_mean))
        self.threshold.append(np.mean(threshold_mean))
        self.f1.append(np.mean(F1_mean))

        # Check if average evaluation loss dropped
        self.early_stopping(f1_score=np.mean(F1_mean))
