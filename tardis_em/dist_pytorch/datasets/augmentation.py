#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
from typing import Optional, Tuple, Union

import numpy as np
import tifffile.tifffile as tiff
from sklearn.neighbors import KDTree, NearestNeighbors

from tardis_em.utils.errors import TardisError
from tardis_em.utils.load_data import ImportDataFromAmira
from tardis_em.utils.normalization import RescaleNormalize, SimpleNormalize


def preprocess_data(
    coord: Union[str, np.ndarray],
    image: Optional[str] = None,
    size: Optional[int] = None,
    include_label=True,
    normalization="simple",
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Data augmentation function.

    Given any supported coordinate file, the function process it with optional image
    data. If image data is used, the image output is a list of flattened image patches
    of a specified size.
    Additionally, the graph output can be created.

    Args:
        coord (str, ndarray): Directory for the file containing coordinate data.
        image (str, None): Directory to the supported image file.
        size (int, None): Image patch size.
        include_label (bool): If True output coordinate array with label ids.
        normalization (str): Type of image normalization.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Returns coordinates and optionally
        graph patch list.
    """

    """ Collect Coordinates [Length x Dimension] """
    if not isinstance(coord, np.ndarray):
        if coord[-4:] == ".csv":
            coord_label = np.genfromtxt(coord, delimiter=",")
            if str(coord_label[0, 0]) == "nan":
                coord_label = coord_label[1:, :]
        elif coord[-4:] == ".npy":
            coord_label = np.load(coord)
        elif coord[-3:] == ".am":
            if image is None:
                amira_import = ImportDataFromAmira(src_am=coord)
                coord_label = amira_import.get_segmented_points()
            else:
                if image.endswith(".am"):
                    amira_import = ImportDataFromAmira(src_am=coord, src_img=image)
                    coord_label = amira_import.get_segmented_points()
                else:
                    amira_import = ImportDataFromAmira(src_am=coord)
                    coord_label = amira_import.get_segmented_points()
    else:
        coord_label = coord.copy()

    if coord_label.shape[1] not in [3, 4]:
        TardisError(
            "",
            "tardis_em/dist_pytorch/dataset/augmentation.py",
            f"Coord file {coord} is without labels."
            "Expected dim to be in [3, 4] for 2D or 3D "
            f"coord but got {coord_label.shape[1]}",
        )

    """ Coordinates without labels """
    coords = coord_label[:, 1:]

    if size is None:
        size = 1

    if coords.shape[1] == 3:
        size = (size, size, size)
    else:
        size = (size, size)

    """ Collect Image Patches [Channels x Length] """
    # Normalize image between 0,1
    if image is not None:
        if normalization not in ["simple", "minmax", None]:
            TardisError(
                "124",
                "tardis_em/dist_pytorch/dataset/augmentation.py",
                f"Not implemented normalization. Given {normalization} "
                "But expected simple or minmax!",
            )

        if normalization == "simple":
            normalization = SimpleNormalize()
        elif normalization == "rescale":
            normalization = RescaleNormalize()

    if image is not None and coord[-3:] != ".am":
        # Crop images size around coordinates
        img_stack = tiff.imread(image)

        crop_tiff = Crop2D3D(
            image=img_stack, size=size, normalization=normalization
        )  # Z x Y x X

        # Load images in an array
        if len(size) == 2:
            img = np.zeros((len(coords), size[0] * size[1]))

            for i in range(img.shape[0]):
                point = coords[i]  # X x Y
                img[i, :] = np.array(crop_tiff(center_point=point)).flatten()

        elif len(size) == 3:
            img = np.zeros((len(coords), size[0] * size[1] * size[2]))

            for i in range(img.shape[0]):
                point = coords[i]  # X x Y x Z
                img[i, :] = np.array(crop_tiff(center_point=point)).flatten()

    elif image is not None and image.endswith(".am"):
        """Collect Image Patches for .am binary files"""
        img_stack, _ = amira_import.get_image()

        # Crop image around coordinates
        crop_tiff = Crop2D3D(
            image=img_stack, size=size, normalization=normalization
        )  # Z x Y x X

        # Load images patches into an array
        if len(size) == 2:
            img = np.zeros((len(coords), size[0] * size[1]))
        else:
            img = np.zeros((len(coords), size[0] * size[1] * size[2]))

        for i in range(img.shape[0]):
            point = coords[i]  # X x Y x Z
            img[i, :] = np.array(crop_tiff(center_point=point)).flatten()
    else:
        img = np.zeros(size)

    """ If not Include label build graph """
    if include_label:
        return coord_label, img
    else:
        graph_builder = BuildGraph()
        graph = graph_builder(coord=coord_label)

        return coords, img, graph


class BuildGraph:
    """
    GRAPH REPRESENTATION FROM 2D/3D COORDINATES

    The main class is to build a graph representation of any given point cloud based on
    the labeling information and optionally point distances.

    The BuildGraph class outputs a 2D array of a graph representation built for the
    filament-like structure which allows for only a maximum of 2 connections per node.
    Or for an object structure where the cap interaction per node limit was fixed as 4.
    The graph representation for a mesh-like object is computed by identifying
    all points in the class and searching for 4 KNN for each node inside the class.

    Args:
        K (int): Number of maximum connections per node.
    """

    def __init__(self, K=2, mesh=False):
        self.K = K
        self.mesh = mesh

    def __call__(self, coord: np.ndarray) -> np.ndarray:
        """
        Graph representation builder.

        Assuming the coordinate array is stored in a variable named 'coords',
        where the first column represents the class ID and the remaining three
        columns represent the XYZ coordinates.

        Args:
            coord (np.ndarray): A coordinate array of the shape (Nx[3, 4]).

        Returns:
            np.ndarray: Graph representation 2D array.
        """
        # extract the class ID and XYZ coordinates from the array
        class_id = coord[:, 0]
        xyz = coord[:, 1:]

        # build the connectivity matrix
        N = coord.shape[0]
        graph = np.zeros((N, N))
        if self.mesh:
            # build a NearestNeighbors object for efficient nearest neighbor search
            nn = NearestNeighbors(n_neighbors=self.K, algorithm="kd_tree").fit(xyz)

            # find the indices of the K-nearest neighbors for each point
            _, indices = nn.kneighbors(xyz)

            for i in range(N):
                for j in indices[i]:
                    if class_id[i] == class_id[j]:  # check class ID before adding edges
                        graph[i, j] = 1
                        # graph[j, i] = 1
        else:
            all_idx = np.unique(coord[:, 0])
            for i in all_idx:
                points_in_contour = np.where(coord[:, 0] == i)[0].tolist()

                for j in points_in_contour:
                    # Self-connection
                    graph[j, j] = 1

                    # First point in contour
                    if j == points_in_contour[0]:  # First point
                        if (j + 1) <= (len(coord) - 1):
                            graph[j, j + 1] = 1
                    # Last point
                    elif j == points_in_contour[len(points_in_contour) - 1]:
                        graph[j, j - 1] = 1
                    else:  # Point in the middle
                        graph[j, j + 1] = 1
                        graph[j, j - 1] = 1

                # Check Euclidean distance between fist and last point
                ends_distance = np.linalg.norm(
                    coord[points_in_contour[0]][1:] - coord[points_in_contour[-1]][1:]
                )

                # If < 2 nm pixel size, connect
                if ends_distance < 5:
                    graph[points_in_contour[0], points_in_contour[-1]] = 1
                    graph[points_in_contour[-1], points_in_contour[0]] = 1

        # Ensure self-connection
        np.fill_diagonal(graph, 1)

        return graph


class Crop2D3D:
    """
    2D/3D IMAGE CROPPING

    Center crop gave the image to the specified size.

    Args:
        image (np.ndarray): Image array.
        size (tuple): Uniform cropping size.
        normalization (class): Normalization type.
    """

    def __init__(self, image: np.ndarray, size: tuple, normalization):
        self.image = image
        self.size = size
        self.normalization = normalization

        if len(size) == 2:
            self.width, self.height = image.shape
            self.depth = None
        else:
            self.depth, self.width, self.height = image.shape

    @staticmethod
    def get_xyz_position(
        center_point: int, size: int, max_size: int
    ) -> Tuple[int, int]:
        """
        Given the center point, calculate the range to crop.

        Args:
            center_point (int): XYZ coordinate for center point.
            size (int): Crop size.
            max_size (int): Axis maximum size is used to calculate the offset.

        Returns:
            Tuple[int, int]: Min and max int values refer to the position on the axis.
        """
        x0 = center_point - (size / 2)
        x1 = center_point + (size / 2)

        """ Check if frame feet into image, if not shift it """
        if x0 < 0:
            move_by = abs(x0)
            x0 = x0 + move_by
            x1 = x1 + move_by

        if x1 > max_size:
            move_by = x1 - max_size
            x0 = x0 - move_by
            x1 = x1 - move_by

        return int(x0), int(x1)

    def __call__(self, center_point: tuple) -> np.ndarray:
        """
        Call for image cropping.

        Args:
            center_point (tuple): XYZ coordinates to crop image.

        Returns:
            np.ndarray: Cropped image patch.
        """
        if len(center_point) not in [2, 3]:
            TardisError(
                "113",
                "tardis_em/dist_pytorch/dataset/augmentation.py",
                "Given position for cropping is not 2D or 3D!. "
                f"Given {center_point}. But expected shape in [2, 3]!",
            )
        if len(center_point) != len(self.size):
            TardisError(
                "124",
                "tardis_em/dist_pytorch/dataset/augmentation.py",
                f"Given cropping shape {len(self.size)} is not compatible "
                f"with given cropping coordinates shape {len(center_point)}!",
            )

        if len(center_point) == 3:
            z0, z1 = self.get_xyz_position(
                center_point=center_point[-1], size=self.size[-1], max_size=self.depth
            )
            x0, x1 = self.get_xyz_position(
                center_point=center_point[0], size=self.size[0], max_size=self.height
            )
            y0, y1 = self.get_xyz_position(
                center_point=center_point[1], size=self.size[1], max_size=self.width
            )
            crop_img = self.image[z0:z1, y0:y1, x0:x1]

            if crop_img.shape != (self.size[-1], self.size[0], self.size[1]):
                crop_df = np.array(crop_img)
                shape = crop_img.shape

                crop_img = np.zeros((self.size[2], self.size[0], self.size[1]))
                crop_img[0 : shape[0], 0 : shape[1], 0 : shape[2]] = crop_df
        elif len(center_point) == 2:
            x0, x1 = self.get_xyz_position(
                center_point=center_point[0], size=self.size[0], max_size=self.height
            )
            y0, y1 = self.get_xyz_position(
                center_point=center_point[1], size=self.size[1], max_size=self.width
            )
            crop_img = self.image[y0:y1, x0:x1]

            if crop_img.shape != (self.size[0], self.size[1]):
                crop_df = np.array(crop_img)
                shape = crop_img.shape
                crop_img = np.zeros((self.size[0], self.size[1]))
                crop_img[0 : shape[0], 0 : shape[1]] = crop_df

        if self.normalization is not None:
            return self.normalization(crop_img)
        else:
            return crop_img


def upsample_pc(org_coord: np.ndarray, sampled_coord: np.ndarray):
    """
    upsample_pc _summary_

    Args:
        org_coord (np.ndarray): _description_
        sampled_coord (np.ndarray): _description_
    """
    # build a KDTree for efficient nearest neighbor search
    tree = KDTree(sampled_coord[:, 1:])

    # find the indices of the K-nearest neighbors for each point
    _, indices = tree.query(org_coord[:, 1:], k=2)

    # Take only first NN point
    indices = indices[:, 1, np.newaxis]
    indices = sampled_coord[indices, 0]

    return np.hstack((indices, org_coord[:, 1]))
