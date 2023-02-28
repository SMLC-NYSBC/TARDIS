#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################
from os import getcwd
from os.path import join

import numpy as np
import torch
from torch import nn

from tardis.dist_pytorch.utils.segment_point_cloud import GraphInstanceV2
from tardis.utils.metrics import eval_graph_f1, mcov
from tardis.utils.trainer import BasicTrainer


class DistTrainer(BasicTrainer):
    """
    DIST MODEL TRAINER
    """

    def __init__(self,
                 **kwargs):
        super(DistTrainer, self).__init__(**kwargs)

        self.node_input = self.structure['node_input']

        self.Graph0_1 = GraphInstanceV2(threshold=0.1, connection=4)
        self.Graph0_5 = GraphInstanceV2(threshold=0.5, connection=4)
        self.Graph0_9 = GraphInstanceV2(threshold=0.9, connection=4)

        self.mCov0_1, self.mCov0_5, self.mCov0_9 = [], [], []

    @staticmethod
    def _update_desc(stop_count: int,
                     metric: list) -> str:
        desc = f'Epochs: early_stop: {stop_count}; F1: [{metric[0]:.2f}/; {metric[1]:.2f}]; ' \
               f'mCov 0.5: [{metric[2]:.2f}/; {metric[3]:.2f}]; ' \
               f'mCov 0.9: [{metric[4]:.2f}/; {metric[5]:.2f}]'
        return desc

    def _update_epoch_desc(self):
        # For each Epoch load be t model from previous run
        if self.id == 0:
            self.epoch_desc = 'Epochs: early_stop: 0; best F1: NaN'
        else:
            self.epoch_desc = self._update_desc(self.early_stopping.counter,
                                                [round(np.max(self.f1), 3),
                                                 round(self.f1[-1:], 3),
                                                 round(np.max(self.mCov0_5), 3),
                                                 round(self.mCov0_5[-1:], 3),
                                                 round(np.max(self.mCov0_9), 3),
                                                 round(self.mCov0_9[-1:], 3)])

    def _save_metric(self) -> bool:
        """ Save training metrics """
        if len(self.training_loss) > 0:
            np.savetxt(join(getcwd(),
                            f'{self.checkpoint_name}_checkpoint', 'training_losses.csv'),
                       self.training_loss, delimiter=';')
        if len(self.validation_loss) > 0:
            np.savetxt(join(getcwd(),
                            f'{self.checkpoint_name}_checkpoint', 'validation_losses.csv'),
                       self.validation_loss, delimiter=',')
        if len(self.f1) > 0:
            np.savetxt(join(getcwd(),
                            f'{self.checkpoint_name}_checkpoint', 'eval_metric.csv'),
                       np.column_stack([self.accuracy, self.precision, self.recall,
                                        self.threshold, self.f1,
                                        self.mCov0_1, self.mCov0_5, self.mCov0_9]),
                       delimiter=',')

        """ Save current model weights"""
        # If mean evaluation loss is higher than save checkpoint
        if all(self.f1[-1:][0] >= i for i in self.f1[:-1]):
            torch.save({
                'model_struct_dict': self.structure,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            },
                join(getcwd(),
                     f'{self.checkpoint_name}_checkpoint',
                     f'{self.checkpoint_name}_checkpoint.pth'))

        torch.save({
            'model_struct_dict': self.structure,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        },
            join(getcwd(),
                 f'{self.checkpoint_name}_checkpoint',
                 'model_weights.pth'))

        if self.early_stopping.early_stop:
            return True
        return False

    def _train(self):
        """
        Run model training.
        """
        # Update progress bar
        self._update_progress_bar(loss_desc='Training: (loss 1.000)', idx=0)

        # Run training for DIST model
        for idx, (e, n, g, _, _) in enumerate(self.training_DataLoader):
            """Mid-training eval"""
            self._mid_training_eval(idx=idx)

            """Training"""
            for edge, node, graph in zip(e, n, g):
                edge, graph = edge.to(self.device), graph.to(self.device)
                self.optimizer.zero_grad()

                if self.node_input:
                    edge = self.model(coords=edge, node_features=node.to(self.device))
                else:
                    edge = self.model(coords=edge, node_features=None)

                # Back-propagate
                loss = self.criterion(edge[:, 0, :], graph)  # Calc. loss
                loss.backward()  # One backward pass
                self.optimizer.step_and_update_lr()  # Update the parameters

                # Store training loss metric
                loss_value = loss.item()
                self.training_loss.append(loss_value)
                lr = self.optimizer._get_lr_scale()

                # Update progress bar
                self._update_progress_bar(loss_desc=f'Training: (loss {loss_value:.4f};'
                                                    f' LR: {lr:.5f})',
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
        mcov0_1, mcov0_5, mcov0_9 = [], [], []

        for idx, (e, n, g, o, _) in enumerate(self.validation_DataLoader):
            coord = [x.cpu().detach().numpy()[0, :] for x in e]

            for edge, node, graph, out in zip(e, n, g, o):
                edge, graph = edge.to(self.device), graph.to(self.device)
                out = out.cpu().detach().numpy()[0, :]

                with torch.no_grad():
                    if self.node_input:
                        node = node.to(self.device)
                        edge = self.model(coords=edge, node_features=node)
                    else:
                        edge = self.model(coords=edge, node_features=None)

                    loss = self.criterion(edge[0, :], graph)

                    acc, prec, recall, f1, th = eval_graph_f1(logits=torch.sigmoid(edge[:, 0, :]),
                                                              targets=graph)

                    target = self.Graph0_5.patch_to_segment(graph=[graph[0, :]],
                                                            coord=coord,
                                                            idx=[out],
                                                            prune=0,
                                                            sort=False)

                    input = torch.where(torch.sigmoid(edge[0, 0, :]) > 0.1, 1, 0)
                    input0_1 = self.Graph0_1.patch_to_segment(graph=[input],
                                                              coord=coord,
                                                              idx=[out],
                                                              prune=0,
                                                              sort=False)
                    input = torch.where(torch.sigmoid(edge[0, 0, :]) > 0.5, 1, 0)
                    input0_5 = self.Graph0_1.patch_to_segment(graph=[input],
                                                              coord=coord,
                                                              idx=[out],
                                                              prune=0,
                                                              sort=False)
                    input = torch.where(torch.sigmoid(edge[0, 0, :]) > 0.9, 1, 0)
                    input0_9 = self.Graph0_1.patch_to_segment(graph=[input],
                                                              coord=coord,
                                                              idx=[out],
                                                              prune=0,
                                                              sort=False)

                # Avg. precision score
                valid_losses.append(loss.item())
                accuracy_mean.append(acc)
                precision_mean.append(prec)
                recall_mean.append(recall)
                F1_mean.append(f1)
                threshold_mean.append(th)
                mcov0_1.append(mcov(input0_1, target))
                mcov0_5.append(mcov(input0_5, target))
                mcov0_9.append(mcov(input0_9, target))

                valid = f'Validation: (loss: {loss.item():.4f}; F1: {f1:.2f}) ' \
                        f'mCov[0.5]: {mcov0_5[:-1]}; mCov[0.9]: {mcov0_9[:-1]}'

                # Update progress bar
                self._update_progress_bar(loss_desc=valid, idx=idx, train=False)

        # Reduce eval. metric with mean
        self.validation_loss.append(np.mean(valid_losses))
        self.accuracy.append(np.mean(accuracy_mean))
        self.precision.append(np.mean(precision_mean))
        self.recall.append(np.mean(recall_mean))
        self.threshold.append(np.mean(threshold_mean))
        self.f1.append(np.mean(F1_mean))

        self.mCov0_1.append(np.mean(mcov0_1))
        self.mCov0_5.append(np.mean(mcov0_5))
        self.mCov0_9.append(np.mean(mcov0_9))

        # Check if average evaluation loss dropped
        self.early_stopping(f1_score=self.f1[-1:][0])


class CDistTrainer(BasicTrainer):
    """
    C_DIST MODEL TRAINER
    """

    def __init__(self,
                 **kwargs):
        super(CDistTrainer, self).__init__(**kwargs)

        if self.structure['dist_type'] == 'semantic':
            self.criterion_cls = nn.CrossEntropyLoss()

    def _train(self):
        """
        Run model training.
        """
        # Update progress bar
        self._update_progress_bar(loss_desc='Training: (loss 1.000)', idx=0)

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
                loss = self.criterion(edge[0, :], graph) + self.criterion_cls(out_cls,
                                                                              cls)
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
                edge, graph, cls = edge.to(self.device), graph.to(self.device), cls.to(self.device)

                with torch.no_grad():
                    if self.node_input:
                        node = node.to(self.device)
                        edge, out_cls = self.model(coords=edge, node_features=node)
                    else:
                        edge, out_cls = self.model(coords=edge, node_features=None)

                    loss = self.criterion(edge[0, :], graph) + self.criterion_cls(out_cls,
                                                                                  cls)

                    edge = torch.sigmoid(edge[:, 0, :])
                    acc, prec, recall, f1, th = eval_graph_f1(logits=edge, targets=graph)

                # Avg. precision score
                valid_losses.append(loss.item())
                accuracy_mean.append(acc)
                precision_mean.append(prec)
                recall_mean.append(recall)
                F1_mean.append(f1)
                threshold_mean.append(th)
                valid = f'Validation: (loss {loss.item():.4f} Prec: {prec:.2f} Rec: {recall:.2f} F1: {f1:.2f})'

                # Update progress bar
                self._update_progress_bar(loss_desc=valid, idx=idx)
        # Reduce eval. metric with mean
        self.validation_loss.append(np.mean(valid_losses))
        self.accuracy.append(np.mean(accuracy_mean))
        self.precision.append(np.mean(precision_mean))
        self.recall.append(np.mean(recall_mean))
        self.threshold.append(np.mean(threshold_mean))
        self.f1.append(np.mean(F1_mean))

        # Check if average evaluation loss dropped
        self.early_stopping(f1_score=self.f1[-1:][0])
