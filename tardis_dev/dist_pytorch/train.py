from os import getcwd
from os.path import join

import numpy as np
import torch
from tardis.utils.metrics import calculate_F1
from tardis_dev.utils.trainer import BasicTrainer
from torch import nn


class DistTrainer(BasicTrainer):
    def __init__(self,
                 **kwargs):
        super(DistTrainer, self).__init__(**kwargs)

        self.node_input = self.structure['node_input']

    def _train(self):
        # Update progress bar
        self._update_progress_bar(loss_value=1.000,
                                  idx=0)

        # Run training for DIST model
        for idx, (e, n, g, _, _) in enumerate(self.training_DataLoader):
            if idx % (len(self.training_DataLoader) // 4) == 0:
                self._validate()
                self.epoch_desc = self._update_desc(self.early_stopping.counter,
                                                    [round(np.max(self.f1), 3),
                                                     self.f1[-1:]])

                # Update checkpoint weights if validation loss dropped
                if all(self.validation_loss[-1:][0] >= i for i in self.validation_loss[:-1]):
                    torch.save({'model_struct_dict': self.structure,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict()},
                               join(getcwd(),
                                    f'{self.checkpoint_name}_checkpoint',
                                    f'{self.checkpoint_name}_checkpoint.pth'))

            for edge, node, graph in zip(e, n, g):
                edge, graph = edge.to(self.device), graph.to(self.device)
                self.optimizer.zero_grad()

                if self.node_input:
                    node = node.to(self.device)
                    out = self.model(coords=edge,
                                     node_features=node)
                else:
                    out = self.model(coords=edge,
                                     node_features=None)

                # Back-propagate
                loss = self.criterion(out[:, 0, :], graph)  # Calc. loss
                loss.backward()  # One backward pass
                self.optimizer.step()  # Update the parameters

                # Store training loss metric
                loss_value = loss.item()
                self.training_loss.append(loss_value)

                # Update progress bar
                self._update_progress_bar(loss_value=f'Training: (loss {loss_value:.4f})',
                                          idx=idx)

    def _validate(self):
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
                        out = self.model(coords=edge,
                                         node_features=node)
                    else:
                        out = self.model(coords=edge,
                                         node_features=None)

                    loss = self.criterion(out[0, :],
                                          graph)
                    out = torch.sigmoid(out[:, 0, :])
                    out = torch.where(out > 0.5, 1, 0)

                acc, prec, recall, f1, th = calculate_F1(logits=out,
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
                self._update_progress_bar(loss_value=valid,
                                          idx=idx)

        # Reduce eval. metric with mean
        self.validation_loss.append(np.mean(valid_losses))
        self.accuracy.append(np.mean(accuracy_mean))
        self.precision.append(np.mean(precision_mean))
        self.recall.append(np.mean(recall_mean))
        self.f1.append(np.mean(F1_mean))


class C_DistTrainer(BasicTrainer):
    def __init__(self,
                 **kwargs):
        super(C_DistTrainer, self).__init__(**kwargs)

        if self.structure['dist_type'] == 'semantic':
            self.criterion_cls = nn.CrossEntropyLoss(reduction='mean')

    def _train(self):
        # Update progress bar
        self._update_progress_bar(loss_value=1.000,
                                  idx=0)

        for idx, (e, n, g, _, c) in enumerate(self.training_DataLoader):
            if idx % (len(self.training_DataLoader) // 4) == 0:
                self._validate()
                self.epoch_desc = self._update_desc(self.early_stopping.counter,
                                                    [round(np.max(self.f1), 3),
                                                     self.f1[-1:]])

                # Update checkpoint weights if validation loss dropped
                if all(self.validation_loss[-1:][0] >= i for i in self.validation_loss[:-1]):
                    torch.save({'model_struct_dict': self.structure,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict()},
                               join(getcwd(),
                                    f'{self.checkpoint_name}_checkpoint',
                                    f'{self.checkpoint_name}_checkpoint.pth'))

            for edge, node, cls, graph in zip(e, n, c, g):
                edge, graph = edge.to(self.device), graph.to(self.device)
                cls = cls.to(self.device)
                self.optimizer.zero_grad()

                if self.node_input:
                    node = node.to(self.device)
                    out, out_cls = self.model(coords=edge,
                                              node_features=node)
                else:
                    out, out_cls = self.model(coords=edge,
                                              node_features=None)

                # Back-propagate
                loss = self.criterion(out[0, :], graph) + \
                    self.criterion_cls(out_cls, cls)
                loss.backward()  # One backward pass
                self.optimizer.step()  # Update the parameters

                # Store evaluation loss metric
                loss_value = loss.item()
                self.training_loss.append(loss_value)

                # Update progress bar
                self._update_progress_bar(loss_value=f'Training: (loss {loss_value:.4f})',
                                          idx=idx)

    def _validate(self):
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
                        out, out_cls = self.model(coords=edge,
                                                  node_features=node)
                    else:
                        out, out_cls = self.model(coords=edge,
                                                  node_features=None)

                    loss = self.criterion(out[0, :], graph) + \
                        self.criterion(out_cls, cls)

                    out = torch.sigmoid(out[:, 0, :])
                    out = torch.where(out > 0.5, 1, 0)

                acc, prec, recall, f1, th = self.calculate_F1(logits=out,
                                                              targets=graph)

                # Avg. precision score
                valid_losses.append(loss.item())
                accuracy_mean.append(acc)
                precision_mean.append(prec)
                recall_mean.append(recall)
                F1_mean.append(f1)
                threshold_mean.append(th)
                valid = f'Validation: (loss {loss.item():.4f} Prec: {prec:.2f} Rec: {recall:.2f} F1: {f1:.2f})'

                # Update progress bar
                self._update_progress_bar(loss_value=valid,
                                          idx=idx)
        # Reduce eval. metric with mean
        self.validation_loss.append(np.mean(valid_losses))
        self.accuracy.append(np.mean(accuracy_mean))
        self.precision.append(np.mean(precision_mean))
        self.recall.append(np.mean(recall_mean))
        self.f1.append(np.mean(F1_mean))
