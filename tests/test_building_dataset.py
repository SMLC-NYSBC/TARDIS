from os import listdir, mkdir
from tardis.slcpy_data_processing.build_training_dataset import BuildDataSet
from os.path import join
from shutil import rmtree

class TestDataSetBuilder:

    def test_main_class(self):
        builder = BuildDataSet(dataset_dir=join('tests', 'test_data', 'data_loader'),
                               circle_size=250,
                               multi_layer=False,
                               tqdm=True)

        assert builder.idx_img == ['am3D.am']
        assert builder.idx_mask == ['am3D.CorrelationLines.am']

    def test_builder(self):
        mkdir(join('tests', 'test_data', 'data_loader', 'train'))
        mkdir(join('tests', 'test_data', 'data_loader', 'train', 'imgs'))
        mkdir(join('tests', 'test_data', 'data_loader', 'train', 'masks'))
        
        builder = BuildDataSet(dataset_dir=join('tests', 'test_data', 'data_loader'),
                               circle_size=250,
                               multi_layer=False,
                               tqdm=True)

        builder.__builddataset__(trim_xy=128,
                                 trim_z=128)
        assert len(listdir(join('tests', 'test_data', 'data_loader', 'train', 'imgs'))) == \
            len(listdir(join('tests', 'test_data', 'data_loader', 'train', 'masks')))

            
        rmtree(join('tests', 'test_data', 'data_loader', 'train'))
