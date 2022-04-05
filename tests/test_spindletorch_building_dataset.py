from os import listdir, mkdir
from os.path import join
from shutil import rmtree

from tardis.utils.utils import BuildTestDataSet
from tardis.slcpy_data_processing.build_training_dataset import BuildTrainDataSet


class TestDataSetBuilder:

    def test_main_class(self):
        builder = BuildTrainDataSet(dataset_dir=join('tests', 'test_data', 'data_loader'),
                                    circle_size=250,
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
                                    multi_layer=False,
                                    tqdm=True)

        builder.__builddataset__(trim_xy=128,
                                 trim_z=128)
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
                                    multi_layer=False,
                                    tqdm=True)

        builder.__builddataset__(trim_xy=128,
                                 trim_z=128)

        test_build = BuildTestDataSet(dataset_dir=join('tests', 'test_data', 'data_loader'),
                                      train_test_ration=10)
        test_build.__builddataset__()

        assert [f'{f[:-4]}_mask.tif' for f in listdir(join('tests', 'test_data', 'data_loader', 'test', 'imgs'))] == \
            listdir(join('tests', 'test_data', 'data_loader', 'test', 'masks'))

        rmtree(join('tests', 'test_data', 'data_loader', 'train'))
        rmtree(join('tests', 'test_data', 'data_loader', 'test'))
