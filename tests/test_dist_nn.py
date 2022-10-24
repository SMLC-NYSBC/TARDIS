import torch
from tardis_dev.dist_pytorch.dist import C_DIST, DIST

# from tardis_dev.utils.aws import get_weights_aws


class TestGraphFormer:

    def rand_tensor(self,
                    shape: tuple):
        return torch.rand(shape)

    # def test_dist_with_aws(self):
    #     # Test for model compatibility with aws weights
    #     weight = torch.load(get_weights_aws(network='dist',
    #                                         subtype='without_img',
    #                                         model='microtubules'),
    #                         map_location='cpu')
    #     model_dict = weight['model_struct_dict']
    #     globals().update(model_dict)

    #     model = DIST(n_out=gf_out,
    #                  node_input=node_input,
    #                  node_dim=gf_node_dim,
    #                  edge_dim=gf_edge_dim,
    #                  num_layers=gf_layers,
    #                  num_heads=gf_heads,
    #                  coord_embed_sigma=gf_sigma,
    #                  dropout_rate=gf_dropout,
    #                  structure=gf_structure,
    #                  predict=False)

    #     model.load_state_dict(weight['model_state_dict'])

    def test_dist_wo_rgb(self):
        for n_dim in [64, 32, 16, None]:
            for e_dim in [64, 32, 16]:
                for n_layer in [6, 3, 1]:
                    for n_head in [8, 4, 2, 1]:
                        model = DIST(n_out=1,
                                     node_input=0,
                                     node_dim=n_dim,
                                     edge_dim=e_dim,
                                     num_layers=n_layer,
                                     num_heads=n_head,
                                     num_cls=None,
                                     dropout_rate=0,
                                     coord_embed_sigma=16,
                                     predict=False)
                        x = model(coords=self.rand_tensor((1, 5, 3)),
                                  node_features=None)
                        assert x.shape == torch.Size((1, 1, 5, 5))

                        x = model(coords=self.rand_tensor((1, 5, 2)),
                                  node_features=None)
                        assert x.shape == torch.Size((1, 1, 5, 5))

    def test_dist_w_rgb(self):
        for n_dim in [64, 32, 16]:
            for e_dim in [64, 32, 16]:
                for n_layer in [6, 3, 1]:
                    for n_head in [8, 4, 2, 1]:
                        model = DIST(n_out=1,
                                     node_input=3,
                                     node_dim=n_dim,
                                     edge_dim=e_dim,
                                     num_layers=n_layer,
                                     num_heads=n_head,
                                     dropout_rate=0,
                                     coord_embed_sigma=16,
                                     predict=False)
                        x = model(coords=self.rand_tensor((1, 5, 3)),
                                  node_features=self.rand_tensor((1, 5, 3)))
                        assert x.shape == torch.Size((1, 1, 5, 5))

                        model = DIST(n_out=1,
                                     node_input=3,
                                     node_dim=n_dim,
                                     edge_dim=e_dim,
                                     num_layers=n_layer,
                                     num_heads=n_head,
                                     dropout_rate=0,
                                     coord_embed_sigma=16,
                                     predict=False)
                        x = model(coords=self.rand_tensor((1, 5, 2)),
                                  node_features=self.rand_tensor((1, 5, 3)))
                        assert x.shape == torch.Size((1, 1, 5, 5))

    def test_cdist_wo_rgb(self):
        for n_dim in [64, 32, 16, None]:
            for e_dim in [64, 32, 16]:
                for n_layer in [6, 3, 1]:
                    for n_head in [8, 4, 2, 1]:
                        model = C_DIST(n_out=1,
                                       node_input=0,
                                       node_dim=n_dim,
                                       edge_dim=e_dim,
                                       num_layers=n_layer,
                                       num_heads=n_head,
                                       dropout_rate=0,
                                       num_cls=200,
                                       coord_embed_sigma=16,
                                       predict=False)
                        x, cls = model(coords=self.rand_tensor((1, 5, 3)),
                                       node_features=None)
                        assert x.shape == torch.Size((1, 1, 5, 5))
                        assert cls.shape == torch.Size((1, 5, 200))

                        x, cls = model(coords=self.rand_tensor((1, 5, 2)),
                                       node_features=None)
                        assert x.shape == torch.Size((1, 1, 5, 5))
                        assert cls.shape == torch.Size((1, 5, 200))

    def test_cdist_w_rgb(self):
        for n_dim in [64, 32, 16]:
            for e_dim in [64, 32, 16]:
                for n_layer in [6, 3, 1]:
                    for n_head in [8, 4, 2, 1]:
                        model = C_DIST(n_out=1,
                                       node_input=3,
                                       node_dim=n_dim,
                                       edge_dim=e_dim,
                                       num_layers=n_layer,
                                       num_heads=n_head,
                                       dropout_rate=0,
                                       num_cls=200,
                                       coord_embed_sigma=16,
                                       predict=False)
                        x, cls = model(coords=self.rand_tensor((1, 5, 3)),
                                       node_features=self.rand_tensor((1, 5, 3)))
                        assert x.shape == torch.Size((1, 1, 5, 5))
                        assert cls.shape == torch.Size((1, 5, 200))

                        x, cls = model(coords=self.rand_tensor((1, 5, 2)),
                                       node_features=self.rand_tensor((1, 5, 3)))
                        assert x.shape == torch.Size((1, 1, 5, 5))
                        assert cls.shape == torch.Size((1, 5, 200))
