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
    DIST MODEL TRAINER
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
        Utility function to update progress bar description.

        Args:
            stop_count (int): Early stop count.
            metric (list): Best f1 and mCov score.

        Returns:
            str: Updated progress bar status.
        """
        desc = (
            f"Epochs: early_stop: {stop_count}; "
            f"F1: [{metric[0]:.2f}; {metric[1]:.2f}]; "
            f"mCov 0.5: [{metric[2]:.2f}; {metric[3]:.2f}]; "
            f"mCov 0.9: [{metric[4]:.2f}; {metric[5]:.2f}]"
        )
        return desc

    def _update_epoch_desc(self):
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
        """Save training metrics"""
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
        Run model training.
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
        Test model against validation dataset.
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
    DIST MODEL TRAINER
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
        Utility function to update progress bar description.

        Args:
            stop_count (int): Early stop count.
            metric (list): Best f1 and mCov score.

        Returns:
            str: Updated progress bar status.
        """
        desc = (
            f"Epochs: early_stop: {stop_count}; "
            f"F1: [{metric[0]:.2f}; {metric[1]:.2f}]; "
            f"mCov 0.5: [{metric[2]:.2f}; {metric[3]:.2f}]; "
            f"mCov 0.9: [{metric[4]:.2f}; {metric[5]:.2f}]"
        )
        return desc

    def _update_epoch_desc(self):
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
        """Save training metrics"""
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
        Run model training.
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
        node = dict()
        for j in range(len(graph)):
            df = [[id_, i] for id_, i in enumerate(graph[j]) if i >= th and id != j]
            node[j] = dict(df)
        coord_list = dict()
        seg_id = 0
        full_list = list(range(len(graph)))

        while len(full_list) != 0:
            id_, old_id = [full_list[0]], [full_list[0]]
            added = list(node[id_[0]].keys())
            id_ = id_ + added

            while len(id_) != len(old_id):
                old_id = id_
                added_new = []
                for key in added:
                    added_new = added_new + list(node[key].keys())
                added = list(dict.fromkeys(added_new))

                id_ = list(dict.fromkeys(id_ + added))
            id_ = sorted(id_)

            coord_list[seg_id] = id_
            seg_id += 1

            for i in id_:
                full_list.remove(i)

        list_c = []
        for key in coord_list:
            value = coord[coord_list[key], :]
            id_ = np.repeat(key, len(value)).reshape(-1, 1)
            list_c.append(np.hstack((id_, value)))
        return np.concatenate(list_c)

    def _validate(self):
        """
        Test model against validation dataset.
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
    C_DIST MODEL TRAINER
    """

    def __init__(self, **kwargs):
        super(CDistTrainer, self).__init__(**kwargs)

        if self.structure["dist_type"] == "semantic":
            self.criterion_cls = nn.CrossEntropyLoss()

    def _train(self):
        """
        Run model training.
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
        Test model against validation dataset.
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
