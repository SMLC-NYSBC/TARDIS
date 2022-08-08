from os import listdir, mkdir
from os.path import join
from shutil import rmtree

from tardis.slcpy.build_training_dataset import BuildTrainDataSet
from tardis.utils.utils import BuildTestDataSet


class TestDataSetBuilder:

    def test_main_class(self):
        builder = BuildTrainDataSet(dataset_dir=join('tests', 'test_data', 'data_loader'),
                                    circle_size=250,
                                    resize_pixel_size=92,
                                    multi_layer=False,
                                    tqdm=True)

        assert builder.idx_img == ['am3D.am']
        assert builder.idx_mask == ['am3D.CorrelationLines.am']

    def test_train_builder(self):
        mkdir(join('tests', 'test_data', 'data_loader', 'train'))
        mkdir(join('tests', 'test_data', 'data_loader', 'train', 'imgs'))
        mkdir(join('tests', 'test_data', 'data_loader', 'train', 'masks'))

        builder = BuildTrainDataSet(dataset_dir=join('tests', 'test_data', 'data_loader'),
                                    circle_size=250,
                                    resize_pixel_size=92,
                                    multi_layer=False,
                                    tqdm=True)

        builder.__builddataset__(trim_xy=64,
                                 trim_z=64)
        assert len(listdir(join('tests', 'test_data', 'data_loader', 'train', 'imgs'))) == \
            len(listdir(join('tests', 'test_data', 'data_loader', 'train', 'masks')))

        rmtree(join('tests', 'test_data', 'data_loader', 'train'))

    def test_test_builder(self):
        mkdir(join('tests', 'test_data', 'data_loader', 'train'))
        mkdir(join('tests', 'test_data', 'data_loader', 'train', 'imgs'))
        mkdir(join('tests', 'test_data', 'data_loader', 'train', 'masks'))
        mkdir(join('tests', 'test_data', 'data_loader', 'test'))
        mkdir(join('tests', 'test_data', 'data_loader', 'test', 'imgs'))
        mkdir(join('tests', 'test_data', 'data_loader', 'test', 'masks'))

        builder = BuildTrainDataSet(dataset_dir=join('tests', 'test_data', 'data_loader'),
                                    circle_size=1250,
                                    resize_pixel_size=23.2,
                                    multi_layer=False,
                                    tqdm=True)

        builder.__builddataset__(trim_xy=64,
                                 trim_z=64)

        test_build = BuildTestDataSet(dataset_dir=join('tests', 'test_data', 'data_loader'),
                                      train_test_ration=10,
                                      prefix='_mask')
        test_build.__builddataset__()

        assert [f'{f[:-4]}_mask.tif' for f in
                sorted(listdir(join('tests', 'test_data', 'data_loader', 'test', 'imgs')))] == \
            sorted(listdir(join('tests', 'test_data', 'data_loader', 'test', 'masks')))

        rmtree(join('tests', 'test_data', 'data_loader', 'train'))
        rmtree(join('tests', 'test_data', 'data_loader', 'test'))
