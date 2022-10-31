import numpy as np
import torch
from tardis.utils.metrics import calculate_F1
from tardis.utils.trainer import BasicTrainer
from torch import nn


class DistTrainer(BasicTrainer):
    """
    DIST MODEL TRAINER
    """

    def __init__(self,
                 **kwargs):
        super(DistTrainer, self).__init__(**kwargs)

        self.node_input = self.structure['node_input']

    def _train(self):
        """
        Run model training.
        """
        # Update progress bar
        self._update_progress_bar(loss_desc='Training: (loss 1.000)',
                                  idx=0)

        # Run training for DIST model
        for idx, (e, n, g, _, _) in enumerate(self.training_DataLoader):
            """Mid-training eval"""
            self._mid_training_eval(idx=idx)

            """Training"""
            for edge, node, graph in zip(e, n, g):
                edge, graph = edge.to(self.device), graph.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                if self.node_input:
                    node = node.to(self.device)
                    edge = self.model(coords=edge,
                                      node_features=node)
                else:
                    edge = self.model(coords=edge,
                                      node_features=None)

                # Back-propagate
                loss = self.criterion(edge[:, 0, :], graph)  # Calc. loss
                loss.backward()  # One backward pass
                self.optimizer.step()  # Update the parameters

                # Store training loss metric
                loss_value = loss.item()
                self.training_loss.append(loss_value)

                # Update progress bar
                self._update_progress_bar(loss_desc=f'Training: (loss {loss_value:.4f})',
                                          idx=idx)

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

        for idx, (e, n, g, _, _) in enumerate(self.validation_DataLoader):
            for edge, node, graph in zip(e, n, g):
                edge, graph = edge.to(self.device), graph.to(self.device)

                with torch.no_grad():
                    if self.node_input:
                        node = node.to(self.device)
                        edge = self.model(coords=edge,
                                          node_features=node)
                    else:
                        edge = self.model(coords=edge,
                                          node_features=None)

                    loss = self.criterion(edge[0, :],
                                          graph)
                    edge = torch.sigmoid(edge[:, 0, :])

                acc, prec, recall, f1, th = calculate_F1(logits=edge,
                                                         targets=graph,
                                                         best_f1=True)

                # Avg. precision score
                valid_losses.append(loss.item())
                accuracy_mean.append(acc)
                precision_mean.append(prec)
                recall_mean.append(recall)
                F1_mean.append(f1)
                threshold_mean.append(th)
                valid = f'Validation: (loss {loss.item():.4f} Prec: {prec:.2f} Rec: {recall:.2f} F1: {f1:.2f})'

                # Update progress bar
                self._update_progress_bar(loss_desc=valid,
                                          idx=idx)

        # Reduce eval. metric with mean
        self.validation_loss.append(np.mean(valid_losses))
        self.accuracy.append(np.mean(accuracy_mean))
        self.precision.append(np.mean(precision_mean))
        self.recall.append(np.mean(recall_mean))
        self.threshold.append(np.mean(threshold_mean))
        self.f1.append(np.mean(F1_mean))

        # Check if average evaluation loss dropped
        self.early_stopping(f1_score=self.f1[-1:][0])


class C_DistTrainer(BasicTrainer):
    """
    C_DIST MODEL TRAINER
    """

    def __init__(self,
                 **kwargs):
        super(C_DistTrainer, self).__init__(**kwargs)

        if self.structure['dist_type'] == 'semantic':
            self.criterion_cls = nn.CrossEntropyLoss(reduction='mean')

    def _train(self):
        """
        Run model training.
        """
        # Update progress bar
        self._update_progress_bar(loss_desc='Training: (loss 1.000)',
                                  idx=0)

        for idx, (e, n, g, _, c) in enumerate(self.training_DataLoader):
            """Mid-training eval"""
            self._mid_training_eval(idx=idx)

            for edge, node, cls, graph in zip(e, n, c, g):
                edge, graph = edge.to(self.device), graph.to(self.device)
                cls = cls.to(self.device)
                self.optimizer.zero_grad()

                if self.node_input:
                    node = node.to(self.device)
                    edge, out_cls = self.model(coords=edge,
                                               node_features=node)
                else:
                    edge, out_cls = self.model(coords=edge,
                                               node_features=None)

                # Back-propagate
                loss = self.criterion(edge[0, :], graph) + \
                    self.criterion_cls(out_cls, cls)
                loss.backward()  # One backward pass
                self.optimizer.step()  # Update the parameters

                # Store evaluation loss metric
                loss_value = loss.item()
                self.training_loss.append(loss_value)

                # Update progress bar
                self._update_progress_bar(loss_desc=f'Training: (loss {loss_value:.4f})',
                                          idx=idx)

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
                edge, graph, cls = edge.to(self.device), \
                    graph.to(self.device), \
                    cls.to(self.device)

                with torch.no_grad():
                    if self.node_input:
                        node = node.to(self.device)
                        edge, out_cls = self.model(coords=edge,
                                                   node_features=node)
                    else:
                        edge, out_cls = self.model(coords=edge,
                                                   node_features=None)

                    loss = self.criterion(edge[0, :], graph) + \
                        self.criterion(out_cls, cls)

                    edge = torch.sigmoid(edge[:, 0, :])\

                acc, prec, recall, f1, th = calculate_F1(logits=edge,
                                                         targets=graph,
                                                         best_f1=True)

                # Avg. precision score
                valid_losses.append(loss.item())
                accuracy_mean.append(acc)
                precision_mean.append(prec)
                recall_mean.append(recall)
                F1_mean.append(f1)
                threshold_mean.append(th)
                valid = f'Validation: (loss {loss.item():.4f} Prec: {prec:.2f} Rec: {recall:.2f} F1: {f1:.2f})'

                # Update progress bar
                self._update_progress_bar(loss_desc=valid,
                                          idx=idx)
        # Reduce eval. metric with mean
        self.validation_loss.append(np.mean(valid_losses))
        self.accuracy.append(np.mean(accuracy_mean))
        self.precision.append(np.mean(precision_mean))
        self.recall.append(np.mean(recall_mean))
        self.threshold.append(np.mean(threshold_mean))
        self.f1.append(np.mean(F1_mean))

        # Check if average evaluation loss dropped
        self.early_stopping(f1_score=self.f1[-1:][0])
