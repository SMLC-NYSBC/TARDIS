from os import listdir
from os.path import join
from shutil import move
from typing import Optional


def cal_node_input(patch_size: Optional[tuple] = None):
    """
    FUNCTION TO CALCULATE NUMBER OF NODE INPUTS BASED ON IMAGE PATCH SIZE

    Args:
        patch_size: Image patch size
    """
    if patch_size is not None:
        n_dim = len(patch_size)
        node_input = 1

        for i in range(n_dim):
            node_input = node_input * patch_size[i]

        return node_input
    else:
        return None


def BuildTrainDataSet(dir: str,
                      coord_format: tuple,
                      with_img: bool,
                      img_format: Optional[tuple] = None):
    """
    STANDARD BUILDER FOR TRAINING DATASETS

    Args:
        dir: Directory where the file should outputted
        coord_format: Format of the coordinate files
        with_img: If True, expect corresponding image files
        img_format: Allowed format that can be used

    Returns:
        _type_: _description_
    """
    assert len([f for f in listdir(dir) if f.endswith(coord_format)]) > 0, \
        f'No file found in given dir {dir}'
    file_format = []

    idx_coord = [f for f in listdir(dir) if f.endswith(coord_format)]

    for i in idx_coord:
        move(src=join(dir, i),
             dst=join(dir, 'train', 'masks', i))
    file_format.append([f for f in coord_format if idx_coord[0].endswith(f)][0])

    """Sort coord with images if included"""
    if with_img:
        assert len([f for f in listdir(dir) if f.endswith(img_format)]) > 0, \
            f'No file found in given dir {dir}'

        idx_coord = [f for f in listdir(dir) if f.endswith(img_format)]

        for i in idx_coord:
            move(src=join(dir, i),
                 dst=join(dir, 'train', 'imgs', i))
        file_format.append([f for f in img_format if idx_coord[0].endswith(f)][0])

    return file_format
