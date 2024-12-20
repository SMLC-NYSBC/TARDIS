#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
from collections import defaultdict
from itertools import combinations
from typing import List, Optional, Tuple, Union

from scipy.spatial import KDTree

try:
    import open3d as o3d
except ModuleNotFoundError:
    pass
from tardis_em.utils import SCANNET_COLOR_MAP_20, rgb_color

import matplotlib.pyplot as plt
import numpy as np


def img_is_color(img):
    """
    Determines if a given image is colored or grayscale by examining its color
    channels. The function checks whether all three color channels (red, green,
    and blue) are identical. If they are identical, the image is considered
    grayscale.

    :param img: A NumPy array representing an image. The input must have three
        dimensions (height, width, channels). The first two dimensions represent
        the image height and width, while the third dimension represents the
        color channels.
    :type img: numpy.ndarray
    :return: A boolean value. Returns `True` if the image is grayscale (all
        color channels are identical), `False` otherwise.
    :rtype: bool
    """
    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if np.all(c1 == c2) and np.all(c2 == c3):
            return True

    return False


def show_image_list(
    list_images,
    list_titles=None,
    list_cmaps=None,
    list_mask_cmaps=None,
    grid=True,
    num_cols=2,
    figsize=(20, 10),
    title_fontsize=30,
    list_masks=None,
    dpi=100,
):
    """
    Displays a list of images in a grid format, with optional customization for titles,
    colormaps, masks, and layout settings. Each image can be individually styled by
    providing corresponding lists for titles, colormaps, or masks.

    :param list_images: List of numpy arrays representing the images to be displayed.
    :param list_titles: Optional list of titles corresponding to each image.
    :param list_cmaps: Optional colormap setting for images. Can be a single colormap
        (str) applied to all images, or a list specifying a colormap for each image.
    :param list_mask_cmaps: Optional colormap setting for overlay masks. Can be a
        single colormap (str) applied to all masks, or a list specifying a colormap
        for each mask.
    :param grid: Boolean flag to enable or disable gridlines over images.
    :param num_cols: Number of columns in the grid layout.
    :param figsize: Tuple specifying the size of the overall figure in inches.
    :param title_fontsize: Font size for the optional titles for each subplot.
    :param list_masks: Optional list of masks that overlay the corresponding images.
        Each mask should have the same dimensions as the corresponding image.
    :param dpi: Resolution of the figure in dots per inch.
    :return: None.
    """
    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), "%d imgs != %d titles" % (
            len(list_images),
            len(list_titles),
        )

    if list_cmaps is not None:
        assert isinstance(list_cmaps, (list, str))
        if not isinstance(list_cmaps, str):
            assert len(list_images) == len(list_cmaps), "%d imgs != %d cmaps" % (
                len(list_images),
                len(list_cmaps),
            )

    if list_masks is not None:
        assert len(list_masks) == len(list_images)

    num_images = len(list_images)
    num_cols = min(num_images, num_cols)
    num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, dpi=dpi)

    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):
        img = list_images[i]
        if list_masks is not None:
            mask = list_masks[i]

        if isinstance(list_cmaps, str):
            cmap = list_cmaps
        else:
            cmap = (
                list_cmaps[i]
                if list_cmaps is not None
                else (None if img_is_color(img) else "gray")
            )

        if list_mask_cmaps is not None:
            if isinstance(list_mask_cmaps, str):
                cmap_mask = list_mask_cmaps
            else:
                cmap_mask = (
                    list_mask_cmaps[i] if list_mask_cmaps is not None else "Reds"
                )
        else:
            cmap_mask = "Reds"

        list_axes[i].imshow(img, cmap=cmap)
        if list_masks is not None:
            list_axes[i].imshow(mask, cmap=cmap_mask, alpha=0.5)

        if title_fontsize is not None:
            list_axes[i].set_title(
                list_titles[i] if list_titles is not None else "Image %d" % i,
                fontsize=title_fontsize,
            )
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    plt.show()


def _dataset_format(coord: np.ndarray, segmented: bool) -> Tuple[np.ndarray, bool]:
    """
    Formats the dataset coordinate array based on the dimensionality and segmentation status.
    This function ensures that input `coord` is in appropriate dimensions (2D or 3D) by fixing
    the missing dimensions when necessary, depending on whether the dataset is segmented or not.
    Additionally, it validates the compatibility of the input dimensions and outputs a
    boolean flag indicating the validation status.

    :param coord: A numpy array containing coordinates of the dataset. Should be 2D or 3D
        with or without labels depending on the segmentation status.
    :type coord: np.ndarray
    :param segmented: A boolean flag indicating whether the dataset is segmented or not.
        If True, the `coord` is expected to contain labels for each data point.
    :type segmented: bool
    :return: A tuple containing a transformed numpy array and a boolean flag.
        The first element is the corrected `coord` dataset and the second element
        indicates whether the input data passed validation checks (`True` or `False`).
    :rtype: Tuple[np.ndarray, bool]
    """
    check = True

    if segmented:
        if coord.shape[1] not in [3, 4]:
            check = False
            print("Coord data must be 2D/3D with labels (4D/5D)")

        # Correct 2D to 3D
        if coord.shape[1] == 3:
            coord = np.vstack(
                (coord[:, 0], coord[:, 1], coord[:, 2], np.zeros((coord.shape[0],)))
            ).T
    else:
        if coord.shape[1] not in [2, 3]:
            check = False
            print("Coord data must be 2D/3D with labels (2D/3D)")

        # Correct 2D to 3D
        if coord.shape[1] == 2:
            coord = np.vstack((coord[:, 0], coord[:, 1], np.zeros((coord.shape[0],)))).T

    return coord, check


def _rgb(
    coord: np.ndarray, segmented: bool, ScanNet=False, color=False, filaments=False
) -> np.ndarray:
    """
    Generates an RGB color representation based on input coordinates and optional parameters
    such as segmented, ScanNet compatibility, color flag, and filament processing. The function
    produces color mapping for each unique identifier in the input coordinates array. Depending
    on the options provided, it can assign random colors, predefined ScanNet colors, or default
    red for non-segmented data. It handles additional functionality for filament-specific
    processing if activated.

    :param coord: The input ndarray containing data, where the first column typically represents
        unique IDs to map to RGB colors. The size and format of the array depend on the specific
        input data.
    :param segmented: Boolean flag to indicate whether the input data is segmented. Segmented
        data triggers unique RGB assignment to distinct segments or IDs.
    :param ScanNet: Optional boolean flag. If True and `segmented` is enabled, assigns predefined
        RGB values from SCANNET_COLOR_MAP_20 to each unique ID. Defaults to False.
    :param color: Optional boolean flag. If True, creates an initial uniform zero RGB list for
        unique IDs when `segmented` is enabled. If False, assigns random RGB values to the unique
        IDs. Defaults to False.
    :param filaments: Boolean flag. If True, processes data specifically for unique filament IDs
        in the coordinates array. Generates a list of random RGB values for each unique filament.
        Defaults to False.

    :return: A numpy array with shape (N, 3), where N is the number of rows in the input
        `coord`. Each row contains the assigned RGB color values for the respective ID in the
        input data. If `filaments` is enabled, returns a list of RGB values for the unique
        elements in `coord`.
    """
    if filaments:
        unique_ids = np.unique(coord[:, 0])
        rgb_list = [
            np.array((np.random.rand(), np.random.rand(), np.random.rand()))
            for _ in unique_ids
        ]
        return rgb_list
    rgb = np.zeros((coord.shape[0], 3), dtype=np.float64)

    if segmented:
        if ScanNet:
            for id_, i in enumerate(coord[:, 0]):
                color = SCANNET_COLOR_MAP_20.get(i, SCANNET_COLOR_MAP_20[0])
                rgb[id_, :] = [x / 255 for x in color]
        else:
            unique_ids = np.unique(coord[:, 0])
            if color:
                rgb_list = [np.array((0, 0, 0)) for _ in unique_ids]
            else:
                rgb_list = [
                    np.array((np.random.rand(), np.random.rand(), np.random.rand()))
                    for _ in unique_ids
                ]
            id_to_rgb = {idx: color for idx, color in zip(unique_ids, rgb_list)}

            for id_, i in enumerate(coord[:, 0]):
                df = id_to_rgb[i]
                rgb[id_, :] = df
    else:
        rgb[:] = [1, 0, 0]

    return rgb


def segment_to_graph(coord: np.ndarray) -> list:
    """
    Generates a graph list representation from an array of segment coordinates.

    The function accepts a 2D NumPy array containing segments defined by their
    start and end points. For each segment, the function assigns indices to
    nodes and converts the segments into a list of directed edges. The result
    is a list of directed edges, where each edge is a pair of indices referring
    to the connected nodes.

    The graph representation is suitable for use in algorithms like pathfinding
    or connectivity analysis based on segments.

    :param coord: Input 2D array representing segments. Each row corresponds to a
        segment, with columns indicating specific segment attributes (e.g., start
        and end coordinates).
    :type coord: np.ndarray
    :return: A list of directed edges representing the graph. Each edge is a pair
        of indices denoting a connection between nodes.
    :rtype: list
    """
    graph_list = []
    stop = 0

    for i in np.unique(coord[:, 0]):
        id_ = np.where(coord[:, 0] == i)[0]
        id_ = coord[id_]

        x = 0  # Iterator checking if current point is a first on in the list
        start = stop
        stop += len(id_)

        if x == 0:
            graph_list.append([start, start + 1])

        length = stop - start  # Number of point in a segment
        for j in range(1, length - 1):
            graph_list.append([start + (x + 1), start + x])

            if j != (stop - 1):
                graph_list.append([start + (x + 1), start + (x + 2)])
            x += 1
        graph_list.append([start + (x + 1), start + x])

    return graph_list


def point_cloud_to_mesh(point_cloud, k=6):
    """
    Converts a 3D point cloud into a triangular mesh representation by grouping points
    based on their IDs, constructing a KDTree for nearest neighbor search, and forming
    triangular faces using the k nearest neighbors of each point in the cloud. If a group
    contains fewer than three points, it will be skipped.

    :param point_cloud: A list of points, where each point includes an ID as the first element
        and its 3D coordinates (x, y, z) as the remaining elements.
    :type point_cloud: list[list[float]]

    :param k: The number of nearest neighbors used for forming triangles. Defaults to 6 if not specified.
    :type k: int
    """
    # Initialize lists to store all vertices and faces across IDs
    all_vertices = []
    all_faces = []
    vertex_offset = 0  # To keep track of the current vertex index across different IDs

    # Group points by ID
    point_groups = defaultdict(list)
    for point in point_cloud:
        point_id = point[0]
        coordinates = point[1:]
        point_groups[point_id].append(coordinates)

    for point_id in point_groups.keys():
        points = point_groups[point_id]
        points = np.array(points)
        if len(points) < 3:
            print(f"Not enough points for triangulation with ID {point_id}")
            continue

        # Build a KDTree for finding nearest neighbors
        kdtree = KDTree(points)
        all_faces_set = set()

        # For each point, find the k nearest neighbors and form triangles
        neighbors = kdtree.query(points, k=k)[
            1
        ]  # Get all neighbors at once for efficiency

        # Generate triangles by connecting each point with pairs of neighbors
        for i, neighbor_indices in enumerate(neighbors):
            for j, m in combinations(
                neighbor_indices[1:], 2
            ):  # Avoid using the point itself
                face = tuple(sorted([i, j, m]))
                all_faces_set.add(face)

        # Convert set of faces to a NumPy array
        faces = np.array(list(all_faces_set))

        # Append the points and faces for this ID to the main lists
        all_vertices.append(points)
        all_faces.append(faces)
        vertex_offset += len(points)  # Update offset for the next group

    return all_vertices, all_faces


def rotate_view(vis):
    """
    Rotates the view of the visualization. This function adjusts the render
    options for the visualization object by setting the background color to
    black and enabling the option to show back faces of the mesh. Additionally,
    it rotates the view slightly along the horizontal axis.

    :param vis: Visualization object to manipulate.
    :type vis: open3d.visualization.Visualizer
    :return: False
    :rtype: bool
    """
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.mesh_show_back_face = True

    ctr = vis.get_view_control()
    ctr.rotate(2.0, 0.0)

    return False


def background_view(vis):
    """
    Sets the background color of the visualization to black and enables displaying
    back faces of the mesh in the visualizer. This function modifies the
    RenderOption object associated with the visualizer instance and returns
    a boolean value.

    :param vis: Visualization object for which the background and mesh rendering
        options are to be set.
    :type vis: Visualizer
    :return: Returns False after successfully setting the background color and
        enabling back face rendering for the mesh.
    :rtype: bool
    """
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.mesh_show_back_face = True
    return False


def VisualizePointCloud(
    coord: np.ndarray,
    segmented: bool = True,
    rgb: Optional[np.ndarray] = None,
    animate=False,
    return_b=False,
):
    """
    Visualize a point cloud given its coordinates and optional additional parameters.

    This function prepares and visualizes 3D point cloud data using Open3D. It allows
    for the visualization of segmented and RGB-colored point clouds. Users can also
    choose to animate the visualization and return the prepared Open3D point cloud
    object instead of directly displaying it.

    :param coord: A NumPy array representing the 3D coordinates of the point cloud.
    :param segmented: A boolean flag indicating whether the input coordinates are
        segmented or not. If True, the visualization will process the input as
        segmented data. Default is True.
    :param rgb: Optional parameter specifying RGB color data for the point cloud.
        If an array is provided, it will be used as the RGB data. If a string is
        provided, it must match a predefined color key. If None, default colors
        will be applied. Default is None.
    :param animate: A boolean flag determining if the visualization should
        include an animation. Default is False.
    :param return_b: A boolean flag indicating whether to return the generated
        Open3D PointCloud object. If True, the function will return the object
        instead of visualizing immediately. Default is False.

    :return: Returns an Open3D PointCloud object if ``return_b`` is set to True.
        No return otherwise.
    """
    coord, check = _dataset_format(coord=coord, segmented=segmented)

    if check:
        pcd = o3d.geometry.PointCloud()

        if segmented:
            pcd.points = o3d.utility.Vector3dVector(coord[:, 1:])
        else:
            pcd.points = o3d.utility.Vector3dVector(coord)

        if rgb is None and coord.shape[1] == 3:
            rgb = rgb_color["red"]
        elif isinstance(rgb, str):
            assert (
                rgb in rgb_color.keys()
            ), f"Color: {rgb} suppoerted. Choose one of: {rgb_color}"
            rgb = rgb_color[rgb]

        if rgb is None:
            pcd.colors = o3d.utility.Vector3dVector(_rgb(coord, True))
        else:
            pcd.paint_uniform_color(rgb)
        if return_b:
            return pcd

        VisualizeCompose(animate=animate, pcd=pcd)


def VisualizeFilaments(
    coord: np.ndarray,
    animate=True,
    with_node=False,
    filament_color=None,
    return_b=False,
):
    """
    Visualizes filament structures based on the provided coordinates and additional options.

    This function takes 3D coordinate data, processes it into a visualizable format, and optionally
    renders it with animations along with node visualizations. It supports configurable filament colors
    and offers the ability to return the visualized LineSet for further use.

    :param coord: The 3D coordinate data array representing the filament geometry. Must follow the
        segmented dataset format.
    :param animate: A boolean flag to indicate whether the visualization should include animation.
        Defaults to True.
    :param with_node: A boolean indicating whether to include node points in the visualization.
        Defaults to False.
    :param filament_color: A string representing the color of the filament, or None to use the default
        white color. The string must correspond to one of the supported color keys in `rgb_color`.
    :param return_b: A boolean specifying whether to return the visualized LineSet object instead of
        rendering it. Defaults to False.
    :return: Returns an Open3D `LineSet` object representing the visualized filament if `return_b` is
        True. Otherwise, no return value.
    :rtype: Optional[o3d.geometry.LineSet]
    """
    coord, check = _dataset_format(coord=coord, segmented=True)

    if filament_color is None:
        filament_color = rgb_color["white"]
    elif isinstance(filament_color, str):
        assert (
            filament_color in rgb_color.keys()
        ), f"Color: {filament_color} suppoerted. Choose one of: {rgb_color}"
        filament_color = rgb_color[filament_color]

    if check:
        if with_node:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coord[:, 1:])
            pcd.colors = o3d.utility.Vector3dVector(_rgb(coord, True))

        graph = segment_to_graph(coord=coord)
        line_set = o3d.geometry.LineSet()

        line_set.points = o3d.utility.Vector3dVector(coord[:, 1:])
        line_set.lines = o3d.utility.Vector2iVector(graph)

        line_set.paint_uniform_color(filament_color)

        if return_b:
            return line_set

        if with_node:
            VisualizeCompose(animate=animate, pcd=pcd, line_set=line_set)
        else:
            VisualizeCompose(animate=animate, line_set=line_set)


def VisualizeScanNet(coord: np.ndarray, segmented: True, return_b=False):
    """
    Visualizes a ScanNet point cloud dataset with the ability to handle segmented and
    non-segmented data. The method can be used to generate and visualize 3D point
    clouds with or without colors based on segmentation. Optionally, the point cloud
    can be returned instead of directly visualizing it.

    :param coord: The input coordinate array representing the point cloud data. The
                  shape and structure depend on whether the data is segmented or
                  non-segmented.
    :param segmented: A boolean indicating whether the input data is segmented. If
                      True, the coordinates include segment information that
                      modifies the visualization process.
    :param return_b: A boolean specifying if the function should return the processed
                    point cloud object. If False, the point cloud is visualized
                    directly without returning the object.

    :return: Optionally returns an Open3D point cloud object if `return_b` is set to
             True.
    """
    coord, check = _dataset_format(coord=coord, segmented=segmented)

    if check:
        pcd = o3d.geometry.PointCloud()

        if segmented:
            pcd.points = o3d.utility.Vector3dVector(coord[:, 1:])
        else:
            pcd.points = o3d.utility.Vector3dVector(coord)
        pcd.colors = o3d.utility.Vector3dVector(_rgb(coord, segmented, True))

        if return_b:
            return pcd

        VisualizeCompose(animate=False, meshes=pcd)


def VisualizeSurface(
    vertices: Union[tuple, list, np.ndarray] = None,
    triangles: Union[tuple, list, np.ndarray] = None,
    point_cloud=None,
    animate=False,
    return_b=False,
):
    """
    Visualizes a 3D surface using vertex and triangle data or a point cloud. The function
    supports optional animation and returns processed meshes when specified.

    :param vertices: The vertices of the surface. Can be a tuple, list, or numpy ndarray.
        If not provided, must provide triangles or point_cloud.
    :param triangles: The triangles composing the surface. Can be a tuple, list, or numpy ndarray.
        If not provided, must provide vertices or point_cloud.
    :param point_cloud: A point cloud array. If provided, vertices and triangles will be
        generated from the point cloud data.
    :param animate: Boolean. If True, animates the composed visualization.
    :param return_b: Boolean. If True, returns the generated mesh objects.
    :return: A list of Open3D TriangleMesh objects if `return_b` is set to True.
    """
    if vertices is None and triangles is None and point_cloud is None:
        return

    if isinstance(vertices, np.ndarray):
        vertices = [vertices]

    if isinstance(triangles, np.ndarray):
        triangles = [triangles]

    if point_cloud is not None:
        if point_cloud.shape[1] == 4:
            pc = []
            for id_ in np.unique(point_cloud[:, 0]):
                pc.append(point_cloud[point_cloud[:, 0] == id_, 1:])
        else:
            pc = [point_cloud]

        vertices, triangles = [], []
        for i in pc:
            pcd = VisualizePointCloud(i, segmented=False, return_b=True)
            pcd.estimate_normals()
            # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, linear_fit=True)[0]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, scale=1
            )[0]

            vertices.append(mesh.vertices)
            triangles.append(mesh.triangles)

    meshes = []
    for v, t in zip(vertices, triangles):
        meshes.append(o3d.geometry.TriangleMesh())

        meshes[-1].vertices = o3d.utility.Vector3dVector(v)
        meshes[-1].triangles = o3d.utility.Vector3iVector(t)
        meshes[-1].paint_uniform_color(list(np.random.random(3)))

        # meshes[-1].filter_smooth_laplacian(5, )
        voxel_size = max(meshes[-1].get_max_bound() - meshes[-1].get_min_bound()) / 512
        meshes[-1] = meshes[-1].simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average,
        )

        meshes[-1].compute_vertex_normals()

    if return_b:
        return meshes

    VisualizeCompose(animate, meshes=meshes)


def VisualizeCompose(animate=False, **kwargs):
    """
    Visualize a collection of 3D objects with optional animation.

    This function takes a collection of 3D objects, processes them,
    and visualizes them using appropriate functionalities. The visualization
    can optionally include animation, triggered based on the `animate` parameter.

    :param animate: A boolean indicating whether to enable animation during
                    visualization.
    :param kwargs: A dictionary of keyword arguments containing 3D objects
                   to be visualized. Values in the dictionary can be either
                   lists of objects or individual 3D objects.
    :return: None
    """
    if all(value is None for value in kwargs.values()):
        return

    objects_ = []
    for o in kwargs.values():
        if isinstance(o, List):
            for i in o:
                objects_.append(i)
        else:
            objects_.append(o)

    if animate:
        o3d.visualization.draw_geometries_with_animation_callback(objects_, rotate_view)
    else:
        o3d.visualization.draw_geometries_with_animation_callback(
            objects_, background_view
        )
