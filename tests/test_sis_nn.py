from tardis.sis_graphformer.graphformer.network import CloudToGraph
from tardis.sis_graphformer.utils.utils import cal_node_input
import torch


class TestGraphFormer:

    def rand_tensor(self,
                          shape: tuple):
        return torch.rand(shape)

    def test_nn_wo_img(self):
        for n_dim in [256, 128, 64, 32, 16]:
            for e_dim in [256, 128, 64, 32, 16]:
                for n_layer in [6, 3, 1]:
                    for n_head in [8, 4, 2, 1]:
                        model = CloudToGraph(n_out=1,
                                             node_input=None,
                                             node_dim=n_dim,
                                             edge_dim=e_dim,
                                             num_layers=n_layer,
                                             num_heads=n_head,
                                             dropout_rate=0,
                                             coord_embed_sigma=16,
                                             predict=False)
                        model(coords=self.rand_tensor((1, 5, 3)),
                              node_features=None,
                              padding_mask=None)
                        model(coords=self.rand_tensor((1, 5, 2)),
                              node_features=None,
                              padding_mask=None)

    def test_nn_w_img(self):
        for n_dim in [256, 128, 64, 32, 16]:
            for e_dim in [256, 128, 64, 32, 16]:
                for n_layer in [6, 3, 1]:
                    for n_head in [8, 4, 2, 1]:
                        model = CloudToGraph(n_out=1,
                                             node_input=cal_node_input((32, 32, 32)),
                                             node_dim=n_dim,
                                             edge_dim=e_dim,
                                             num_layers=n_layer,
                                             num_heads=n_head,
                                             dropout_rate=0,
                                             coord_embed_sigma=16,
                                             predict=False)
                        model(coords=self.rand_tensor((1, 5, 3)),
                              node_features=self.rand_tensor((1, 5, 32768)),
                              padding_mask=None)

                        model = CloudToGraph(n_out=1,
                                             node_input=cal_node_input((32, 32)),
                                             node_dim=n_dim,
                                             edge_dim=e_dim,
                                             num_layers=n_layer,
                                             num_heads=n_head,
                                             dropout_rate=0,
                                             coord_embed_sigma=16,
                                             predict=False)
                        model(coords=self.rand_tensor((1, 5, 2)),
                              node_features=self.rand_tensor((1, 5, 1024)),
                              padding_mask=None)
