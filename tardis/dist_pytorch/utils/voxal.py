from itertools import compress
from typing import Optional

import numpy as np
import torch
from scipy.spatial import distance
from tardis.dist_pytorch.utils.augmentation import BuildGraph


class VoxalizeDataSetV2:
    """
    CLASS TO REDUCE DATASET TO OVERLAPPING VOXAL

    Build voxal grid from the given point cloud and output
    coordinates/graph/image in each voxal as a list of numpy arrays

    Coordinates in each voxal can be downsample if they reach pre-defined
    threshold. In that case, voxal downsampling from open3d library is used to
    downsample point cloud as close as possible to defined threshold and output
    points from the original dataset that are closes to the downsample points
    in order to preserve coordinates.

    Args:
        coord: Coordinates dataset in a shape [Dim, X, Y, (Z)].
            where Dim are segment ids
        image: Image dataset in a shape of [Channels x Length].
        downsampling_threshold: Max number of point in voxal
        downsampling_rate: Downsampling voxal size for open3d voxal_downsampling
        init_voxal_size: Initial voxal size used for voxalization of the point
            cloud. This voxal size will be 'optimize' until each voxal contains
            number of points below downsampling_threshold
        drop_rate: Voxal size optimization drop rate factor.
        graph: If True, graph is output, as coord in shape [Dim, X, Y, (Z)]
            else [X, Y, (Z)]
        tensor: If True return all output as Tensors
    Usage:
        VD = VoxalizeDataSetV2(...
                               graph=True)
        coord, img, graph = VD.voxalize_dataset()

        VD = VoxalizeDataSetV2(...
                               graph=False)
        coord, img = VD.voxalize_dataset()
    """

    def __init__(self,
                 coord: np.ndarray,
                 image: Optional[np.ndarray] = None,
                 downsampling_threshold=500,
                 downsampling_rate: Optional[float] = None,
                 init_voxal_size=10000,
                 drop_rate=1,
                 graph=True,
                 tensor=True):
        # Global data setting
        if graph:
            assert coord.shape[1] in [3, 4], 'If graph True, coord must by of shape' \
                '[Dim x X x Y x (Z)]'
            self.segments_id = coord
            self.coord = coord[:, 1:]
        else:
            assert coord.shape[1] in [2, 3], 'If graph True, coord must by of shape' \
                '[X x Y x (Z)]'
            self.segments_id = None
            self.coord = coord
        self.image = image
        self.torch_output = tensor
        self.graph_output = graph

        # Point cloud downsampling setting
        self.downsampling_threshold = downsampling_threshold
        self.downsampling_rate = downsampling_rate

        # Voxal setting
        self.voxal_patch_size = init_voxal_size
        self.size_expand = init_voxal_size * 0.25  # 25% of voxal size
        self.voxal_stride = init_voxal_size * 0.15  # 10% of voxal size
        self.drop_rate = drop_rate

    def boundary_box(self):
        """
        DEFINE BOUNDARY BOX FOR 2D OR 3D COORD
        """
        box_dim = self.coord.shape[1]

        if box_dim in [2, 3]:
            min_x = np.min(self.coord[:, 0]) - self.size_expand
            max_x = np.max(self.coord[:, 0]) + self.size_expand

            min_y = np.min(self.coord[:, 1]) - self.size_expand
            max_y = np.max(self.coord[:, 1]) + self.size_expand
        if box_dim == 3 and np.min(self.coord[:, 2]) != 0:
            min_z = np.min(self.coord[:, 2]) - self.size_expand
            max_z = np.max(self.coord[:, 2]) + self.size_expand
        else:
            min_z, max_z = 0, 0

        return np.array([(min_x, min_y, min_z),
                        (max_x, max_y, max_z)])

    def voxal_centers(self,
                      boundary_box: np.ndarray):
        """
        SEARCH FOR CENTER OF EACH 2D/3D VOXAL

        Args:
            boundary_box: Coordinate boundary box
        """
        voxal_positions_x = []
        voxal_positions_y = []

        bb_min = boundary_box[0]
        bb_max = boundary_box[1]

        if len(bb_min) == 3:
            z_mean = bb_max[2] / 2
        else:
            z_mean = 0

        # Find X positions for voxals
        x_pos = bb_min[0] + (self.voxal_patch_size / 2)
        voxal_positions_x.append(x_pos)
        while bb_max[0] > x_pos:
            x_pos = x_pos + self.voxal_patch_size
            voxal_positions_x.append(x_pos)

        # Find Y positions for voxal
        y_pos = bb_min[1] + (self.voxal_patch_size / 2)
        voxal_positions_y.append(y_pos)
        while bb_max[1] > y_pos:
            y_pos = y_pos + self.voxal_patch_size
            voxal_positions_y.append(y_pos)

        # Bind X and Y voxal positions
        voxal = []
        voxal_positions_x = voxal_positions_x[::2]
        voxal_positions_y = voxal_positions_y[::2]
        for i in voxal_positions_x:
            voxal.append(np.vstack(([i] * len(voxal_positions_y),
                                    voxal_positions_y,
                                    [z_mean] * len(voxal_positions_y))).T)

        return np.vstack(voxal)

    def points_in_voxal(self,
                        voxal_center: np.ndarray):
        """
        BOOLEAN INDEXING FOR FILTERING POINT CLOUD IN VOXAL

        Args:
            voxal_center: Numpy array [3, 1] with voxal center coordinate
        """
        voxal_size = self.voxal_patch_size + self.voxal_stride

        coord_idx = (self.coord[:, 0] <= (voxal_center[0] + voxal_size)) & \
                    (self.coord[:, 0] >= (voxal_center[0] - voxal_size)) & \
                    (self.coord[:, 1] <= (voxal_center[1] + voxal_size)) & \
                    (self.coord[:, 1] >= (voxal_center[1] - voxal_size))

        return coord_idx

    def voxal_downsampling(self,
                           coord: np.ndarray):
        """
        VOXAL DOWNSAMPLE MODULE
        The module downsample point cloud based on voxal method then finds
        nearest point in the input cloud which are outputted as down-sampled PC.

        Args:
            coord: Coordinates of points found in voxal
        """
        if self.downsampling_rate is not None:
            from open3d import geometry, utility

            # Downsampling
            pcd = geometry.PointCloud()
            pcd.points = utility.Vector3dVector(coord)
            pcd = np.asarray(pcd.voxel_down_sample(self.downsampling_rate).points)

            idx_ds = []
            dist_matrix = distance.cdist(pcd, coord, 'euclidean')

            # Find point closest to original position
            for i in dist_matrix:
                idx = np.where(i == np.min(i))[0]
                idx_ds.append(idx[0])

            # Build list of point idx to keep
            full_idx = list(range(0, coord.shape[0], 1))
            idx_bool = [True if id in idx_ds else False for id in full_idx]
        else:

            # Build list of point idx to keep
            full_idx = list(range(0, coord.shape[0], 1))
            idx_bool = [True for p in full_idx]

        return idx_bool

    def collect_voxal_idx(self,
                          voxals):
        """
        SELECT NOT EMPTY VOXAL WITH UNIQUE SET OF POINTS

        Args:
            voxals: XYZ coord array of voxal centers
        """
        not_empty_voxal = []
        points_no = []
        for i, voxal in enumerate(voxals):
            # Pick a points idx
            idx = self.points_in_voxal(voxal_center=voxal)

            # Select points from full list
            if self.coord.shape[1] == 2:
                coord_voxal = np.hstack((np.array([0] * self.coord[idx, :].shape[0])[:, None],
                                        self.coord[idx, :]))
            else:
                coord_voxal = self.coord[idx, :]

            if coord_voxal.shape[0] > 1:
                not_empty_voxal.append(i)
                points_no.append(coord_voxal.shape[0])

        return not_empty_voxal, points_no

    def optimize_voxal_size(self):
        """
        Build voxals

        Function search for the optimal voxals based on the number of points

        Return:
            voxals_coord: List of coordinates for the voxals centers
            voxals_idx: List of voxal that contains points
        """
        # Initial check for voxalization
        b_box = self.boundary_box()

        if self.downsampling_rate is not None:
            self.coord = self.coord[self.voxal_downsampling(self.coord), :]

        if self.coord.shape[0] <= self.downsampling_threshold:
            voxal_coord_x = b_box[1][0] - ((abs(b_box[0][0]) + abs(b_box[1][0])) / 2)
            voxal_coord_y = b_box[1][1] - ((abs(b_box[0][1]) + abs(b_box[1][1])) / 2)

            if b_box.shape[1] == 3:
                voxal_coord_z = b_box[1][2] - ((abs(b_box[0][2]) + abs(b_box[1][2])) / 2)
                voxals_coord = [voxal_coord_x, voxal_coord_y, voxal_coord_z]
            else:
                voxals_coord = [voxal_coord_x, voxal_coord_y]

            voxal_idx = [0]

            return voxals_coord, voxal_idx

        # Initial voxalization with self.voxal_patch_size
        if self.voxal_patch_size == 0:
            self.voxal_patch_size = int(np.max(b_box))
        voxal_size = self.voxal_patch_size

        voxals_coord = self.voxal_centers(boundary_box=b_box)
        voxal_idx, piv = self.collect_voxal_idx(voxals=voxals_coord)

        # Optimize voxal size based on no_point threshold
        break_if = 0

        drop_rate = self.drop_rate
        while not all(i <= self.downsampling_threshold for i in piv):
            self.voxal_patch_size = self.voxal_patch_size - self.drop_rate

            if self.voxal_patch_size <= 0:
                break_if += 1

                self.drop_rate = drop_rate / 2
                self.voxal_patch_size = voxal_size - self.drop_rate

            if break_if == 3:
                print('Could not find valid voxal size, prediction of full point cloud!')
                return [voxals_coord[0]], [voxal_idx[0]]

            self.size_expand = self.voxal_patch_size * 0.25
            self.voxal_stride = self.voxal_patch_size * 0.15

            voxals_coord = self.voxal_centers(boundary_box=self.boundary_box())
            voxal_idx, piv = self.collect_voxal_idx(voxals=voxals_coord)

        return voxals_coord, voxal_idx

    @staticmethod
    def normalize_idx(coord_with_idx: np. ndarray):
        unique_idx = list(np.unique(coord_with_idx[:, 0]))
        norm_idx = list(range(len(np.unique(coord_with_idx[:, 0]))))

        for id, i in enumerate(unique_idx):
            idx_list = list(np.where(coord_with_idx[:, 0] == i)[0])

            for j in idx_list:
                coord_with_idx[j, 0] = norm_idx[id]

        return coord_with_idx

    def output_format(self,
                      data: np.ndarray):
        if self.torch_output:
            data = torch.from_numpy(data).type(torch.float32)

        return data

    def voxal_patch_size(self):
        return self.voxal_patch_size

    def voxalize_dataset(self,
                         out_idx=True,
                         prune=0):
        """
        Main function used to build voxalized dataset

        Args:
            out_idx: If True, return point id of points in each voxal
            prune: Prune voxal with less then given number of nodes
        """
        # Build voxal dataset
        voxals_centers, voxals_idx = self.optimize_voxal_size()

        coord_voxal = []
        img_voxal = []
        graph_voxal = []
        output_idx = []

        if len(voxals_idx) == 1:
            # Transform 2D coord to 3D of shape [Z, Y, X]
            if self.coord.shape[1] == 2:
                coord_ds = np.vstack((self.coord[:, 0],
                                      self.coord[:, 1],
                                      np.zeros((self.coord.shape[0], )))).T
            else:
                coord_ds = self.coord

            coord_ds = self.voxal_downsampling(coord_ds)

            if self.graph_output:
                coord_label = self.segments_id[coord_ds, :]
                build_graph = BuildGraph(coord=coord_label,
                                         pixel_size=None)
                graph_voxal.append(self.output_format(build_graph()))

            coord_voxal.append(self.output_format(self.coord[coord_ds, :]))
            if self.image is not None:
                img_voxal.append(self.output_format(self.image[coord_ds, :]))
            else:
                img_voxal.append(self.output_format(np.zeros((1, 1))))

            if out_idx:
                output_idx.append(np.where(coord_ds)[0])
        else:
            for i in voxals_idx:
                df_voxal_keep = self.points_in_voxal(voxals_centers[i])

                if self.image is not None:
                    df_img = self.image[df_voxal_keep, :]
                else:
                    df_img = np.zeros((1, 1))
                df_voxal = self.coord[df_voxal_keep, :]
                output_df = np.where(df_voxal_keep)[0]

                # Transform 2D coord to 3D of shape [Z, Y, X]
                if df_voxal.shape[1] == 2:
                    coord_ds = np.vstack((np.zeros((df_voxal.shape[0], )),
                                          df_voxal[:, 1],
                                          df_voxal[:, 0])).T
                else:
                    coord_ds = df_voxal

                coord_ds = self.voxal_downsampling(coord_ds)

                if np.sum(coord_ds) > prune:
                    if self.graph_output:
                        segment_voxal = self.segments_id[df_voxal_keep, :]
                        segment_voxal = self.normalize_idx(segment_voxal[coord_ds, :])
                        build_graph = BuildGraph(coord=segment_voxal,
                                                 pixel_size=None)
                        graph_voxal.append(self.output_format(build_graph()))

                    coord_voxal.append(self.output_format(df_voxal[coord_ds, :]))
                    if self.image is not None:
                        img_voxal.append(self.output_format(df_img[coord_ds, :]))
                    else:
                        img_voxal.append(self.output_format(df_img))

                    if out_idx:
                        output_idx.append(output_df[coord_ds])
                elif not prune:
                    if self.graph_output:
                        segment_voxal = self.segments_id[df_voxal_keep, :]
                        segment_voxal = self.normalize_idx(
                            segment_voxal[coord_ds, :])
                        build_graph = BuildGraph(coord=segment_voxal,
                                                 pixel_size=None)
                        graph_voxal.append(self.output_format(build_graph()))

                    coord_voxal.append(self.output_format(df_voxal[coord_ds, :]))
                    if self.image is not None:
                        img_voxal.append(self.output_format(df_img[coord_ds, :]))
                    else:
                        img_voxal.append(self.output_format(df_img))

                    if out_idx:
                        output_idx.append(output_df[coord_ds])
        # Find voxal that have more then 50% unique points
        unique_idx = [False if np.any([True for k in [len(set(output_idx[id]).intersection(j)) for idx, j in enumerate(output_idx) if idx != id] if k > len(output_idx[id])/1.5]) \
            else True for id, _ in enumerate(output_idx)]

        if self.graph_output:
            return list(compress(coord_voxal, unique_idx)), \
                list(compress(img_voxal, unique_idx)), \
                    list(compress(graph_voxal, unique_idx)), \
                        list(compress(output_idx, unique_idx))
        else:
            return list(compress(coord_voxal, unique_idx)), \
                list(compress(img_voxal, unique_idx)), \
                    list(compress(output_idx, unique_idx))
