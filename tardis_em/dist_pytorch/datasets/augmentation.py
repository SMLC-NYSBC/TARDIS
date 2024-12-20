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
    Preprocesses coordinate and image data for further analysis.

    This function reads and processes geometric coordinate data and associated image data,
    applying normalization and cropping based on specified parameters. It supports various
    file formats for input coordinates (.csv, .npy, .am) and optionally uses the image data
    aligned with these coordinates. The processed output can include labels, graphs, or
    specific image subsets, depending on the provided parameters.

    :param coord: Required input coordinates in either raw array format or file paths
        (strings) pointing to supported formats like .csv, .npy, or .am files. Depending
        on the format, data is loaded and processed for subsequent use.
    :type coord: Union[str, np.ndarray]
    :param image: Optional input image path corresponding to the given coordinates. If
        provided, image patches are extracted and normalized around coordinate positions.
        It can also be a .am file for specific processing.
    :type image: Optional[str]
    :param size: Optional size of image patches to be cropped. If not provided, a
        default size of 1 is used. Determines whether the cropping is applied in 2D
        or 3D, based on the dimensions of the provided coordinates.
    :type size: Optional[int]
    :param include_label: Indicates whether to return labels along with coordinates
        and image patches. If False, an additional graph is created from coordinates.
    :type include_label: bool
    :param normalization: Specifies the normalization method for image data. Accepts
        "simple" or "minmax" as methods or None for no normalization. Default is "simple".
    :type normalization: str
    :return: Returns a tuple that may consist of coordinates with labels, images, or
        a graph depending on the value of `include_label`. If `include_label` is True,
        a tuple of coordinates, labels, and images is returned. Otherwise, a graph of
        coordinates and images is included in the returned tuple.
    :rtype: Union[Tuple[np.ndarray, np.ndarray],
                  Tuple[np.ndarray, np.ndarray, np.ndarray]]
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
    Creates a representation of points as a graph where each edge represents the connectivity between
    points based on their proximity or contour sequence. Depending on the configuration, the graph is
    constructed either through Nearest Neighbors for mesh-based structures or explicit connections for
    points belonging to the same contour. The graph is represented as a binary matrix, with optional
    handling of self-connections.
    """

    def __init__(self, K=2, mesh=False):
        self.K = K
        self.mesh = mesh

    def __call__(self, coord: np.ndarray) -> np.ndarray:
        """
        Builds a graph based on connectivity and distance criteria of input coordinates. This method either
        uses Nearest Neighbors for identifying local connections in a mesh structure or constructs explicit
        connections for contours, all while ensuring self-connections. The graph represents the relationship
        between points using a binary connectivity matrix.

        :param coord: Input array of shape (N, M) where N is the number of points and M is the number of attributes
            per point. The first column corresponds to the class IDs, and the subsequent columns represent
            the XYZ coordinates.
        :type coord: np.ndarray
        :return: A binary adjacency matrix of shape (N, N) where each element indicates the connection status
            between points. A value of 1 indicates a connection, while 0 indicates no connection.
        :rtype: np.ndarray
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
    Class for 2D or 3D image cropping based on a center position and size.

    This class facilitates cropping sections of an image (both 2D and 3D depending
    on the input) by specifying a center position and dimensions for the crop. It
    supports normalization of the cropped section if a normalization function is
    provided.
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
        Calculate the adjusted x-coordinate positions for a frame, ensuring that the
        frame fits within the bounds of a specified maximum size.

        This method adjusts the lower and upper bounds of the frame's position based
        on its size, ensuring it remains within the allowed range (`max_size`). If the
        computed position exceeds boundaries, it shifts the frame appropriately.

        :param center_point: The center point of the frame, around which the lower
            and upper x-coordinate positions are calculated.
        :param size: The size of the frame on the x-axis, which will be distributed
            evenly around the center point.
        :param max_size: The maximum allowable boundary in the x-axis that the frame
            must stay within.
        :return: A tuple containing the adjusted lower and upper x-coordinate
            positions, ensuring the frame lies within the bounds.
        :rtype: Tuple[int, int]
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
        Perform cropping and return the cropped sub-image based on the specified center point.

        This method uses the provided center point coordinates and the predefined crop size to
        extract a sub-image from the original image. It handles both 2D and 3D cropping scenarios
        based on the dimensionality of the center point and size attributes. Additionally, if the
        cropped image does not match the expected size, the method pads it with zeros to ensure
        compatibility. Optional normalization is applied to the cropped image if a normalization
        function is provided.

        :param center_point: The coordinates representing the center position for cropping.
            The tuple should contain either 2 or 3 values, depending on the dimension of the image
            (2D or 3D).

        :return: The cropped sub-image as a numpy array. If a normalization function is provided,
            the returned sub-image is normalized. If no normalization is defined, the raw cropped
            sub-image is returned.
        :rtype: np.ndarray
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
    Upsample the point cloud by associating each original coordinate (org_coord) with its
    nearest neighbor in the sampled coordinate set (sampled_coord). The function finds the
    nearest neighbor using a KDTree for efficient spatial querying and computes the
    new coordinates based on these associations.

    :param org_coord: A numpy array of shape (n, m), where n is the number of points
                      and m represents the dimensionality of the space.
                      The first column represents unique identifiers,
                      and the remaining columns represent the coordinates of the points.
    :param sampled_coord: A numpy array of shape (k, m), where k is the number of sampled points
                          and m represents the dimensionality of the space.
                          The first column represents unique identifiers,
                          and the remaining columns represent the coordinates of the points.

    :return: A numpy array of shape (n, m-1) representing the upsampled point cloud.
             The first column contains the identifiers of the nearest neighbors from the
             sampled_coord array, and the remaining columns replicate the coordinate
             dimensions of the input org_coord.
    """
    # build a KDTree for efficient nearest neighbor search
    tree = KDTree(sampled_coord[:, 1:])

    # find the indices of the K-nearest neighbors for each point
    _, indices = tree.query(org_coord[:, 1:], k=2)

    # Take only first NN point
    indices = indices[:, 1, np.newaxis]
    indices = sampled_coord[indices, 0]

    return np.hstack((indices, org_coord[:, 1]))
