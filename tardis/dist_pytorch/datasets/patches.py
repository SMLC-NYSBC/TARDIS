#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from typing import Optional, Tuple, Union

import numpy as np
import torch

from tardis.dist_pytorch.datasets.augmentation import BuildGraph
from tardis.utils.errors import TardisError


class PatchDataSet:
    """
    BUILD PATCHED DATASET

    Class for computing optimal patch size for a maximum number of points per patch.
    The optimal size of the patch is determined by max number of points. It works by first
    checking if 'init_patch_size' != 0, if True then 'init_patch_size' is set as
    max size that contain the whole point cloud. Then class optimizes the size of
    'init_patch_size' by sequentially dropping it by a set 'drop_rate' value. This
    action is continue till 'init_patch_size' is < 0 or, class found 'init_patch_size'
    where all computed patches have a number of points below the threshold.

    Patches are computed building voxel 2D/3D grid of the size given by 'init_patch_size'
    In the end, patches with a smaller number of points are marge with their neighbor
    in a way that will respect 'max_number_of_points' policy.

    Output is given as a list of arrays as torch.Tensor or np.ndarray.

    Args:
        label_cls (np.ndarray, None): Optional class id array for each point in the
            point cloud.
        rgb (np.ndarray, None): Optional RGB feature array for each point in the point
            cloud.
        patch_3d (bool): If True, compute patches in 3D. If False, patches are
            computed in 2D and if coord (N, 3), Z dimension is np.inf.
        max_number_of_points (int): Maximum allowed a number of points per patch.
        init_patch_size (float): Initial patch size. If 0, the initial patch size
            is taken as the highest value from the computed boundary box.
        drop_rate (float): Optimizer step size for reducing the size of patches.
        graph (bool): If True output computed graph for each patch of point cloud.
        tensor (bool): If True output all datasets as torch.Tensor.
    """

    def __init__(self,
                 label_cls=None,
                 rgb=None,
                 patch_3d=False,
                 max_number_of_points=500,
                 init_patch_size=0,
                 drop_rate=1,
                 graph=True,
                 tensor=True):
        # Global data setting
        self.label_cls = label_cls
        self.rgb = rgb
        self.segments_id = None
        self.coord = None

        # Point cloud down-sampling setting
        self.DOWNSAMPLING_TH = max_number_of_points

        # Patch setting
        self.TORCH_OUTPUT = tensor
        self.GRAPH_OUTPUT = graph
        self.PATCH_3D = patch_3d
        self.init_patch_size = init_patch_size
        self.INIT_PATCH_SIZE = init_patch_size
        self.expand = 0.025
        self.EXPAND = 0.025  # Expand boundary box by 2.5%
        self.stride = 0.15
        self.STRIDE = 0.15  # Create 15% overlaps between patches

        if init_patch_size == 0:
            self.SIZE_EXPAND = self.EXPAND
            self.PATCH_STRIDE = self.STRIDE
        else:
            self.SIZE_EXPAND = init_patch_size * self.EXPAND
            self.PATCH_STRIDE = init_patch_size * self.STRIDE

        self.drop_rate = drop_rate
        self.DROP_RATE = drop_rate

    def _init_parameters(self):
        # Patch setting
        self.INIT_PATCH_SIZE = self.init_patch_size
        if self.init_patch_size == 0:
            self.EXPAND = self.expand
            self.STRIDE = self.stride
            self.SIZE_EXPAND = self.EXPAND
            self.PATCH_STRIDE = self.STRIDE
        else:
            self.SIZE_EXPAND = self.init_patch_size * self.expand
            self.PATCH_STRIDE = self.init_patch_size * self.stride

    def _boundary_box(self) -> np.ndarray:
        """
        Utile class function to compute boundary box in 2D or 3D

        Returns:
            np.ndarray: Boundary box dimensions
        """
        box_dim = self.coord.shape[1]

        # Define x,y and (z) min and max sizes
        if box_dim in [2, 3]:
            min_x = np.min(self.coord[:, 0]) - self.SIZE_EXPAND
            max_x = np.max(self.coord[:, 0]) + self.SIZE_EXPAND

            min_y = np.min(self.coord[:, 1]) - self.SIZE_EXPAND
            max_y = np.max(self.coord[:, 1]) + self.SIZE_EXPAND
        if box_dim == 3 and np.min(self.coord[:, 2]) != 0:
            min_z = np.min(self.coord[:, 2]) - self.SIZE_EXPAND
            max_z = np.max(self.coord[:, 2]) + self.SIZE_EXPAND
        else:
            min_z, max_z = 0, 0

        return np.array([(min_x, min_y, min_z),
                        (max_x, max_y, max_z)])

    def _collect_patch_idx(self,
                           patches: np.ndarray) -> Tuple[list, list]:
        """
        Utile class function to compute patch metrics.

        Args:
            patches (np.ndarray): List of all patch centers.

        Returns:
            Tuple[list, list]: ID's list of all patches with more than 1 point
            and list of point numbers in each patch.
        """
        not_empty_patch = []
        points_no = []

        for i, patch in enumerate(patches):
            # Pick a points idx's
            idx = self._points_in_patch(patch_center=patch)

            # Select points from full point cloud array
            if self.coord.shape[1] == 2:
                coord_patch = np.hstack((np.array([0] *
                                                  self.coord[idx, :].shape[0])[:, None],
                                        self.coord[idx, :]))
            else:
                coord_patch = self.coord[idx, :]

            # Evaluate patch for number of points if it contains more then 1 point
            if coord_patch.shape[0] > 1:
                not_empty_patch.append(i)
                points_no.append(coord_patch.shape[0])

        return not_empty_patch, points_no

    @staticmethod
    def _normalize_idx(coord_with_idx: np.ndarray) -> np.ndarray:
        """
        Utile class function to replace ids with ordered output ID values for
        each point in patches. In other words, it produces a standardized ID for each
        point so it can be identified with the source.

        Args:
            coord_with_idx (np.ndarray): Coordinate id value i.
        Returns:
            np.ndarray: An array all points in a patch with corrected ID value.
        """
        unique_idx = list(np.unique(coord_with_idx[:, 0]))
        norm_idx = list(range(len(np.unique(coord_with_idx[:, 0]))))

        for id, i in enumerate(unique_idx):
            idx_list = list(np.where(coord_with_idx[:, 0] == i)[0])

            for j in idx_list:
                coord_with_idx[j, 0] = norm_idx[id]

        return coord_with_idx

    def _output_format(self,
                       data: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        """
        Utile class function to output array in the correct format (numpy or tensor).

        Args:
            data (np.ndarray): Input data for format change.

        Returns:
            np.ndarray: Array in file format specified by self.torch_output.
        """
        if self.TORCH_OUTPUT:
            data = torch.from_numpy(data).type(torch.float32)

        return data

    def _patch_centers(self,
                       boundary_box: np.ndarray) -> np.ndarray:
        """
        Utile class function to compute patches given stored patch size and
        boundary box to output center coordinate for all possible overlapping
        patches.

        Args:
            boundary_box (np.ndarray): Computer point cloud boundary box.

        Returns:
            np.ndarray: Array with XYZ coordinates to localize patch centers.
        """
        patch = []
        patch_positions_x = []
        patch_positions_y = []

        bb_min = boundary_box[0]
        bb_max = boundary_box[1]

        if len(bb_min) == 3:
            z_mean = bb_max[2] / 2
        else:
            z_mean = 0

        # Find X positions for patches
        x_pos = bb_min[0] + (self.INIT_PATCH_SIZE / 2)
        patch_positions_x.append(x_pos)

        while bb_max[0] > x_pos:
            x_pos = x_pos + self.INIT_PATCH_SIZE - self.PATCH_STRIDE
            patch_positions_x.append(x_pos)

        # Find Y positions for patch
        y_pos = bb_min[1] + (self.INIT_PATCH_SIZE / 2)
        patch_positions_y.append(y_pos)
        while bb_max[1] > y_pos:
            y_pos = y_pos + self.INIT_PATCH_SIZE - self.PATCH_STRIDE
            patch_positions_y.append(y_pos)

        # Bind X and Y patch positions
        patch_positions_x = patch_positions_x[::2]
        patch_positions_y = patch_positions_y[::2]

        # Find Z position for patch
        if not self.PATCH_3D:  # Get 3D patches. Z position is center of bb
            for i in patch_positions_x:
                patch.append(np.vstack(([i] * len(patch_positions_y),
                                        patch_positions_y,
                                        [z_mean] * len(patch_positions_y))).T)
        else:  # Get 3D patches. Z position is computed as X and Y position
            patch_positions_z = []

            z_pos = bb_min[2] + (self.INIT_PATCH_SIZE / 2)
            patch_positions_z.append(z_pos)

            while bb_max[2] > z_pos:
                z_pos = z_pos + self.INIT_PATCH_SIZE - self.PATCH_STRIDE
                patch_positions_z.append(z_pos)

            for i in patch_positions_x:
                for j in patch_positions_z:
                    patch.append(np.vstack(([i] * len(patch_positions_y),
                                            patch_positions_y,
                                            [j] * len(patch_positions_y))).T)

        return np.vstack(patch)

    def _points_in_patch(self,
                         patch_center: np.ndarray) -> tuple:
        """
        Utile class function for filtering point cloud and output only point in
        patch.

        Args:
            patch_center (np.ndarray): Array (1, 3) for the given patch center.

        Returns:
            tuple(bool): Array of all points that are enclosed in the given patch.
        """
        patch_size = self.INIT_PATCH_SIZE + self.PATCH_STRIDE

        coord_idx = (self.coord[:, 0] <= (patch_center[0] + patch_size)) & \
                    (self.coord[:, 0] >= (patch_center[0] - patch_size)) & \
                    (self.coord[:, 1] <= (patch_center[1] + patch_size)) & \
                    (self.coord[:, 1] >= (patch_center[1] - patch_size))

        return coord_idx

    def optimize_patch_size(self) -> Union[Tuple[list, list], Tuple[np.ndarray, list]]:
        """
        Main class function to compute optimal patch size.

        The function takes init stored variable and iteratively searches for patch size
        small enough that allow for all patches to have an equal or less max number
        of points.
        """
        """ Initial check for patches """
        b_box = self._boundary_box()

        if self.coord.shape[0] <= self.DOWNSAMPLING_TH:
            patch_coord_x = b_box[1][0] - ((abs(b_box[0][0]) + abs(b_box[1][0])) / 2)
            patch_coord_y = b_box[1][1] - ((abs(b_box[0][1]) + abs(b_box[1][1])) / 2)

            if b_box.shape[1] == 3:
                patch_coord_z = b_box[1][2] - ((abs(b_box[0][2]) + abs(b_box[1][2])) / 2)
                patches_coord = [patch_coord_x, patch_coord_y, patch_coord_z]
            else:
                patches_coord = [patch_coord_x, patch_coord_y]

            patch_idx = [0]

            return patches_coord, patch_idx

        # Initial patronization with self.INIT_PATCH_SIZE
        if self.INIT_PATCH_SIZE == 0:
            self.INIT_PATCH_SIZE = np.max(b_box)
            patch_size = self.INIT_PATCH_SIZE
            self.PATCH_STRIDE = patch_size * self.STRIDE

        patches_coord = self._patch_centers(boundary_box=b_box)
        patch_idx, piv = self._collect_patch_idx(patches=patches_coord)

        # Optimize patch size based on no_point threshold
        break_if = 0

        drop_rate = self.DROP_RATE
        while not all(i <= self.DOWNSAMPLING_TH for i in piv):
            self.INIT_PATCH_SIZE = self.INIT_PATCH_SIZE - self.DROP_RATE

            if self.INIT_PATCH_SIZE <= 0:
                break_if += 1

                self.DROP_RATE = drop_rate / 2
                self.INIT_PATCH_SIZE = patch_size - self.DROP_RATE

            if break_if == 3:
                print('Could not find valid patch size, prediction of full point cloud!')
                return [patches_coord[0]], [patch_idx[0]]

            self.SIZE_EXPAND = self.INIT_PATCH_SIZE * self.EXPAND
            self.PATCH_STRIDE = self.INIT_PATCH_SIZE * self.STRIDE

            patches_coord = self._patch_centers(boundary_box=self._boundary_box())
            patch_idx, piv = self._collect_patch_idx(patches=patches_coord)

        return patches_coord, patch_idx

    def patched_dataset(self,
                        coord: np.ndarray,
                        mesh=False,
                        dist_th: Optional[float] = None) -> Union[Tuple[list, list,
                                                                        list, list, list],
                                                                  Tuple[list, list,
                                                                        list, list]]:
        """
        Main function for processing dataset and return patches.

        Args:
            coord (np.ndarray): 2D or 3D array of the point cloud.
            mesh (boolean): If True, build a graph for meshes, not filaments.
            dist_th (float):  Distance threshold for graph from meshes.

        Returns:
            list[np.ndarray or torch.Tensor]:
                List of arrays (N, 2) or (N, 3) with coordinates of points per patch

                List of an array (N, 3) with RGB value for each point peer patch

                An optional list of all computed graphs from each coord_patch

                List of array (N, 1) with ordered ID value for each point per patch.
                The ordered ID value allows reconstructing point cloud from patches

                List of an array (N, 3) with classes id  for each point peer patch
        """
        coord_patch = []
        graph_patch = []
        output_idx = []

        self._init_parameters()

        if self.GRAPH_OUTPUT:
            assert coord.shape[1] in [3, 4], \
                TardisError('113',
                            'tardis/dist_pytorch/datasets/patches.py',
                            'If graph True, coord must by of shape'
                            f'[Dim x X x Y x (Z)], but is: {coord.shape}')
            self.segments_id = coord
            self.coord = coord[:, 1:]

            graph_builder = BuildGraph(mesh=mesh)
        else:
            assert coord.shape[1] in [2, 3], \
                TardisError('113',
                            'tardis/dist_pytorch/datasets/patches.py',
                            'If graph True, coord must by of shape'
                            f'[X x Y x (Z)], but is: {coord.shape}')
            self.segments_id = None
            self.coord = coord

        if mesh:
            assert dist_th is not None, \
                TardisError('124',
                            'tardis/dist_pytorch/datasets/patches.py',
                            'If mesh, dist_th cannot be None!')

        # Check if point cloud is smaller than max allowed point
        if self.coord.shape[0] <= self.DOWNSAMPLING_TH:
            """ Transform 2D coord to 3D of shape [Z, Y, X] """
            if self.coord.shape[1] == 2:
                coord_ds = np.vstack((self.coord[:, 0],
                                      self.coord[:, 1],
                                      np.zeros((self.coord.shape[0], )))).T
            else:
                coord_ds = self.coord
            coord_ds = [True for _ in list(range(0, coord_ds.shape[0], 1))]

            """ Build point cloud for each patch """
            coord_patch.append(self._output_format(self.coord[coord_ds, :]))

            """ Optionally - Build graph for each patch """
            if self.GRAPH_OUTPUT:
                coord_label = self.segments_id[coord_ds, :]
                coord_label = self._normalize_idx(coord_label)

                if mesh:
                    graph_patch.append(
                        self._output_format(graph_builder(coord=coord_label,
                                                          dist_th=dist_th))
                    )
                else:
                    graph_patch.append(
                        self._output_format(graph_builder(coord=coord_label))
                    )

            """ Build output index for each patch """
            output_idx.append(np.where(coord_ds)[0])

            """ Build class label index for each patch """
            if self.label_cls is not None:
                cls_patch = [self.label_cls]
            else:
                cls_patch = [self._output_format(np.zeros((1, 1)))]

            """ Build rgb node label index for each patch """
            if self.rgb is not None:
                rgb_patch = [self.rgb]
            else:
                rgb_patch = [self._output_format(np.zeros((1, 1)))]
        else:  # Build patches for PC with max num. of point per patch
            """ Find optimal patch centers """
            patches_centers, patches_idx = self.optimize_patch_size()

            all_patch = []
            cls_patch = []
            rgb_patch = []

            """ Find all patches """
            for i in patches_idx:
                all_patch.append(self._points_in_patch(patches_centers[i]))

            """ Combine smaller patches with threshold limit """
            new_patch = []
            while len(all_patch) > 0:
                df = all_patch[0]

                if df.sum() >= self.DOWNSAMPLING_TH:
                    new_patch.append(df)
                    all_patch.pop(0)
                else:
                    while df.sum() <= self.DOWNSAMPLING_TH:
                        if len(all_patch) == 1:
                            break
                        if df.sum() + all_patch[1].sum() > self.DOWNSAMPLING_TH:
                            break
                        df += all_patch[1]
                        all_patch.pop(1)
                    new_patch.append(df)
                    all_patch.pop(0)

            all_patch = new_patch

            """ Build patches """
            for i in all_patch:
                """ Find points and optional images for each patch"""
                df_patch_keep = i

                df_patch = self.coord[df_patch_keep, :]
                output_df = np.where(df_patch_keep)[0]

                # Transform 2D coord to 3D of shape [Z, Y, X]
                if df_patch.shape[1] == 2:
                    coord_ds = np.vstack((np.zeros((df_patch.shape[0], )),
                                          df_patch[:, 1],
                                          df_patch[:, 0])).T
                else:
                    coord_ds = df_patch

                coord_ds = [True for _ in list(range(0, coord_ds.shape[0], 1))]

                """ Build point cloud for each patch """
                coord_patch.append(self._output_format(df_patch[coord_ds, :]))

                """ Optionally - Build graph for each patch """
                if self.GRAPH_OUTPUT:
                    segment_patch = self.segments_id[df_patch_keep, :]
                    segment_patch = self._normalize_idx(segment_patch[coord_ds, :])

                    if mesh:
                        graph_patch.append(
                            self._output_format(graph_builder(coord=segment_patch,
                                                              dist_th=dist_th))
                        )
                    else:
                        graph_patch.append(
                            self._output_format(graph_builder(coord=segment_patch))
                        )

                """ Build output index for each patch """
                output_idx.append(output_df[coord_ds])

                """ Build class label index for each patch """
                if self.label_cls is not None:
                    cls_df = self.label_cls[df_patch_keep]
                    cls_new = np.zeros((cls_df.shape[0], 200))
                else:
                    cls_df = [0]
                    cls_new = np.zeros((1, 200))

                for id, j in enumerate(cls_df):
                    df = np.zeros((1, 200))
                    df[0, int(j)] = 1
                    cls_new[id, :] = df

                cls_patch.append(self._output_format(cls_new))

                """ Build rbg node label index for each patch """
                if self.rgb is not None:
                    rgb_df = self.rgb[df_patch_keep]
                else:
                    rgb_df = np.zeros((1, 1))

                rgb_patch.append(self._output_format(rgb_df))

        if self.GRAPH_OUTPUT:
            return coord_patch, rgb_patch, graph_patch, output_idx, cls_patch
        else:
            return coord_patch, rgb_patch, output_idx, cls_patch
