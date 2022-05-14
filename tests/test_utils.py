import tardis.utils.utils as t_utils
import tardis.spindletorch.utils.utils as sp_utils
import tardis.dist_pytorch.utils.utils as sis_utils
import torch


class TestUtils:
    def test_tardis_utils(self):
        es = t_utils.EarlyStopping(patience=10,
                                   min_delta=0)
        i = 1
        while not es.early_stop:
            i += 1
            es(val_loss=i)
        assert i == 12

    def test_sp_utils(self):
        n = sp_utils.number_of_features_per_level(init_channel_number=64,
                                                  num_levels=5)
        assert n == [64, 128, 256, 512, 1024]

        n = sp_utils.number_of_features_per_level(init_channel_number=32,
                                                  num_levels=6)
        assert n == [32, 64, 128, 256, 512, 1024]

        n = sp_utils.max_number_of_conv_layer(img=torch.rand((64, 64, 64)),
                                              input_volume=64,
                                              max_out=8,
                                              kernel_size=3,
                                              padding=1,
                                              stride=1,
                                              pool_size=2,
                                              pool_stride=2,
                                              first_max_pool=False)
        assert n == 4

    def test_sis_utils(self):
        assert sis_utils.cal_node_input((64, 64, 64, 64)) == 16777216
        assert sis_utils.cal_node_input((64, 64, 64)) == 262144
        assert sis_utils.cal_node_input((64, 64)) == 4096
