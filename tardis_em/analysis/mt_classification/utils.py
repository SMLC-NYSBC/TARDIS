#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
import numpy as np
from sklearn.neighbors import NearestNeighbors


def count_true_groups(bool_list) -> int:
    """
    Counts the number of 'True' groups in a list of booleans. A 'True' group is defined
    as a contiguous sequence of 'True' values, separated by at least one 'False' value
    or the start/end of the list non-'True' regions. This function iterates through the
    given list and counts distinct 'True' groups.

    :param bool_list: A list of boolean values.
    :type bool_list: list[bool]

    :return: The number of 'True' groups identified in the boolean list.
    :rtype: int
    """
    count = 0
    in_group = False

    for value in bool_list:
        if value and not in_group:
            count += 1
            in_group = True
        elif not value:
            in_group = False
    return count


def distances_of_ends_to_surface(
    vertices_n: np.ndarray, pole_n: np.ndarray, ends: np.ndarray, d1_to_surf=False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the distances from specified end points to a surface and optionally
    to a reference pole.

    This function determines the distances of provided end points to the nearest
    vertex on a specified surface and optionally the further distance to a
    prespecified pole. The function makes use of the Nearest Neighbors algorithm
    for distance computation. If the additional parameter `d1_to_surf` is set,
    the calculated distances include intermediate computations involving the
    nearest neighbors of end points on the surface.

    :param vertices_n: Array of coordinates of the vertices defining the surface.
    :param pole_n: Array of coordinates representing the reference pole position.
    :param ends: Array of coordinates of the end points for which distances are to
        be calculated.
    :param d1_to_surf: Boolean flag. If True, distances from the end points to the
        surface include intermediate computations involving the nearest neighbors
        of specified vertices on the surface. Defaults to False.

    :return: A tuple of two distance arrays. The first array represents distances
        computed involving `d1_to_surf` logic if True, otherwise from ends to
        the nearest vertices. The second array contains distances from end points
        to the reference pole.
    """
    knn_v = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(vertices_n)
    knn_e = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(pole_n.reshape(1, -1))

    if d1_to_surf:
        _, i1 = knn_v.kneighbors(ends[:, 1:])
        d1, _ = knn_e.kneighbors(vertices_n[i1.flatten(), :])
    else:
        d1, _ = knn_v.kneighbors(ends[:, 1:])
    d2, _ = knn_e.kneighbors(ends[:, 1:])

    return d1, d2


def distance_to_the_pole(points: np.ndarray, distance_to: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distances between each point and a reference point.

    This function calculates the Euclidean distance from a given set of points
    to a specific reference point. The points and the reference point must be
    provided as numpy arrays. The expected shapes of the input arrays ensure
    that each row in the `points` array represents a single point in the same
    dimensional space as the `distance_to` point.

    :param points: A numpy array of shape (n, m) where `n` is the number of
        points and `m` is the dimensionality of the points.
    :param distance_to: A numpy array of shape (m,) that specifies the
        reference point in the m-dimensional space.

    :return: A numpy array of shape (n,) representing the computed distances
        for each point to the given reference point.
    """
    distances = np.sqrt(np.sum((points - distance_to) ** 2, axis=1))

    return distances


def divide_into_sequences(arr) -> list[list[int]]:
    """
    Divides a list of integers into contiguous subsequences, where each subsequence
    contains consecutive numbers. The method processes the input list and groups
    elements into separate lists based on consecutive relationships.

    :param arr: A list of integers to be divided into sequences. Must be non-empty.
    :type arr: list[int]

    :return: A list of lists, where each internal list represents a contiguous
             subsequence of consecutive integers from the input list.
    :rtype: list[list[int]]
    """
    sequences = []
    current_sequence = [arr[0]]  # Start with the first element

    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + 1:
            # If the current element is consecutive, add it to the current sequence
            current_sequence.append(arr[i])
        else:
            # If not consecutive, finalize the current sequence and start a new one
            sequences.append(current_sequence)
            current_sequence = [arr[i]]

    # Append the last sequence
    sequences.append(current_sequence)

    return sequences


def fill_gaps(float_list: list, n: float) -> np.ndarray:
    """
    Fill gaps between consecutive elements in a list with evenly spaced values.

    This function takes a list of floats and a threshold value, `n`, to determine
    the gap size between consecutive elements. If the gap between two consecutive
    elements is smaller than or equal to `n`, it fills the gap with evenly spaced
    values. The resulting list is then returned as a sorted NumPy array with unique
    values.

    :param float_list: A list of float numbers to process.
    :type float_list: list
    :param n: A float threshold value for determining the maximum gap size.
    :type n: float

    :return: A NumPy array with gaps filled by evenly spaced unique values.
    :rtype: numpy.ndarray
    """
    filled_list = []

    for i in range(len(float_list) - 1):
        filled_list.append(float_list[i])

        # Check if the gap between consecutive elements is larger than n
        if abs(float_list[i + 1] - float_list[i]) <= n:
            num_inserts = int(abs(float_list[i + 1] - float_list[i]))
            step = (float_list[i + 1] - float_list[i]) / (num_inserts + 1)

            # Insert the values into the list
            for j in range(1, num_inserts + 1):
                filled_list.append(int(float_list[i] + step * j))

    # Add the last element
    filled_list.append(float_list[-1])

    return np.unique(filled_list)


def pick_pole_to_surfaces(poles, vertices) -> np.ndarray:
    """
    Determine the order of poles based on their distance to the centroid of provided vertices.

    This function calculates the centroid of the given vertices, computes the Euclidean distance
    of the two poles from the centroid, and orders the poles based on which is closer to the centroid.
    If the second pole is closer to the centroid than the first pole, their positions are swapped
    in the returned array.

    :param poles: A 2D NumPy array with shape (2, N), where each row represents a pole's coordinates.
    :param vertices: A NumPy array with shape (1, M, N), where M represents the number of vertices
        and N represents the dimensions (e.g., 2D, 3D) of each vertex's coordinates.

    :return: A NumPy array containing the reordered poles based on their distance to the centroid
        of the vertices.
    """
    v_centroid = np.mean(vertices[0], axis=0)

    p1_centroid = np.linalg.norm(v_centroid - poles[0, :])
    p2_centroid = np.linalg.norm(v_centroid - poles[1, :])

    if p1_centroid > p2_centroid:
        return np.array([poles[1, :], poles[0, :]])
    else:
        return poles


def points_on_mesh_knn(
    points: np.ndarray, vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the distances of points to a mesh and identifies if the distances are within a
    threshold.

    The function calculates the nearest distances from given points to the provided mesh vertices
    using k-nearest neighbors. It also determines whether the points lie within a computed distance
    based on the mean edge length of the mesh (mean distance between vertices multiplied by 2). The
    output includes the actual distances and a boolean array indicating if the threshold was met.

    :param points: Coordinates of the points that need to be tested against the mesh vertices.
        Expected to be a 2D NumPy array with each row representing a point in space.
    :param vertices: Coordinates of the mesh vertices. Expected to be a 2D NumPy array with each
        row representing a vertex in space.

    :return: A tuple containing two elements:
        - A NumPy array of distances from each point to the nearest mesh vertex.
        - A NumPy boolean array indicating whether each distance falls within the computed threshold.
    """
    knn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(vertices)
    distances, _ = knn.kneighbors(vertices)
    mean_distance = np.mean(distances[:, 1]) * 2

    distances, _ = knn.kneighbors(points)
    distances_bool = distances[:, 0] <= mean_distance

    return distances[:, 0], np.array(distances_bool)


def select_mt_ids_within_bb(
    vertices_n: np.ndarray, mt_ends1: np.ndarray, mt_ends2=None
) -> np.ndarray:
    """
    Finds microtubule (MT) IDs whose end-points lie within a 3D bounding box defined by the vertices.
    If `mt_ends2` is provided, it performs the filtering on this additional set of MT end-points as well.

    :param vertices_n: A 2D numpy array of shape (n, 3) representing the vertices of a 3D bounding box. Each vertex is
        expected to have x, y, and z coordinates.
    :param mt_ends1: A 2D numpy array where each row represents an MT end-point. Column 0 should contain MT IDs, and
        columns 1, 2, and 3 should correspond to x, y, and z coordinates of the end-points, respectively.
    :param mt_ends2: Optional. A 2D numpy array structured similarly to `mt_ends1`. Each row represents an additional
        set of MT end-points. Default is None.

    :return: A 1D numpy array containing unique MT IDs whose end-points lie within the 3D bounding box defined by
        `vertices_`. If `mt_ends2` is provided, the function updates its result set to include filtered IDs from
        `mt_ends2` as well.
    """
    # Vertices 3D bounding box
    min_x, max_x = np.min(vertices_n[:, 0]), np.max(vertices_n[:, 0])
    min_y, max_y = np.min(vertices_n[:, 1]), np.max(vertices_n[:, 1])
    min_z, max_z = np.min(vertices_n[:, 2]), np.max(vertices_n[:, 2])

    def filter_mts(mt_ends, min_, max_, axis):
        mask = (mt_ends[:, axis] >= min_) & (mt_ends[:, axis] <= max_)

        return mt_ends[mask]

    mt_ends1 = filter_mts(mt_ends1, min_x, max_x, axis=1)
    mt_ends1 = filter_mts(mt_ends1, min_y, max_y, axis=2)
    mt_ends1 = filter_mts(mt_ends1, min_z, max_z, axis=3)
    mt_ends_id = np.unique(mt_ends1[:, 0])

    if mt_ends2 is not None:
        mt_ends2 = np.vstack([i for i in mt_ends2 if i[0] in mt_ends_id])
        mt_ends2 = filter_mts(mt_ends2, min_x, max_x, axis=1)
        mt_ends2 = filter_mts(mt_ends2, min_y, max_y, axis=2)
        mt_ends2 = filter_mts(mt_ends2, min_z, max_z, axis=3)
        mt_ends_id = np.unique(mt_ends2[:, 0])

    return mt_ends_id


def assign_filaments_to_poles(filaments, poles) -> tuple[np.ndarray, np.ndarray]:
    """
    Assigns filaments to the closest of two poles based on their minimal distance and
    reverses filament orientation if necessary to ensure the correct assignment.

    The function calculates the minimal distance between each filament endpoint and
    two given poles, and assigns filaments to the pole they are closest to. If the
    start point of a filament is farther from the assigned pole than the endpoint,
    the function flips the filament orientation. The result is a tuple of two arrays,
    where each array contains the filaments assigned to a particular pole.

    :param filaments: A 2D numpy array of filaments where each row represents a
        point in a filament. The first column contains filament IDs, and the
        remaining columns represent the coordinates of the points.
    :param poles: A 2D numpy array of two poles, where each row corresponds to a
        pole, and the columns represent the coordinates of the pole.

    :return: A tuple of two numpy arrays. The first element contains all filaments
        assigned to pole 1, and the second element contains all filaments assigned
        to pole 2. Each array has rows corresponding to filament points, with the
        first column representing the filament ID and the remaining columns
        representing the coordinates of the filament point.
    """
    # Extract filament IDs
    ids = filaments[:, 0]
    unique_ids, index_starts, counts = np.unique(
        ids, return_index=True, return_counts=True
    )

    # Compute end indices of filaments
    end_indices = index_starts + counts - 1

    # Extract start and end points of each filament
    start_points = filaments[index_starts, 1:]
    end_points = filaments[end_indices, 1:]

    filament_endpoints = np.stack([start_points, end_points], axis=1)

    # Calculate distance between endpoints and poles for broadcasting
    differences = (
        filament_endpoints[:, :, np.newaxis, :] - poles[np.newaxis, np.newaxis, :, :]
    )
    distances = np.linalg.norm(differences, axis=-1)

    # Find minimal distances to each pole for each filament
    min_dist_pole1 = np.min(distances[:, :, 0], axis=1)
    min_dist_pole2 = np.min(distances[:, :, 1], axis=1)

    # Determine which pole is closer for each filament
    assigned_poles = np.where(min_dist_pole1 <= min_dist_pole2, 1, 2)

    # Loop over each filament to flip if necessary and assign to the correct pole
    filament_pole1_list = []
    filament_pole2_list = []
    for idx, (start_idx, end_idx) in enumerate(zip(index_starts, end_indices)):
        # Extract the filament points
        filament_points = filaments[start_idx : end_idx + 1]

        # Get the assigned pole index (0 or 1)
        assigned_pole_index = assigned_poles[idx] - 1

        # Get distances from both endpoints to the assigned pole
        dist_start = distances[idx, 0, assigned_pole_index]
        dist_end = distances[idx, 1, assigned_pole_index]

        # If the start point is further from the assigned pole than the end point
        # flip the filament
        if dist_start < dist_end:
            filament_points = filament_points[::-1]

        # Append the filament to the appropriate list
        if assigned_poles[idx] == 1:
            filament_pole1_list.append(filament_points)
        else:
            filament_pole2_list.append(filament_points)

    # Combine the lists into arrays
    filament_pole1 = (
        np.vstack(filament_pole1_list) if filament_pole1_list else np.empty((0, 4))
    )
    filament_pole2 = (
        np.vstack(filament_pole2_list) if filament_pole2_list else np.empty((0, 4))
    )

    return filament_pole1, filament_pole2


def assign_filaments_to_n_poles(filaments, poles):
    """
    Assigns filaments to the closest of any number of poles based on the minimal distance from any of their points
    and reverses filament orientation if the first point is farther from the assigned pole than the last point.

    The function calculates the minimal distance between any point of each filament and each pole,
    assigns filaments to the closest pole, and flips the filament orientation if necessary.
    The result is a list of arrays, one per pole, containing the assigned filaments.

    :param filaments: A 2D numpy array of shape [m, 4] where each row is [ID, x, y, z] for a point in a filament.
                      Multiple points may share the same ID, representing a filament with variable points.
    :param poles: A 2D numpy array of shape [p, 3], where each row contains the [x, y, z] coordinates of a pole.
    :return: A list of p numpy arrays, where the i-th array contains all filaments assigned to pole i.
             Each array has shape [k, 4] with [ID, x, y, z] rows. Empty arrays have shape [0, 4].
    """
    # Extract filament IDs and coordinates
    ids = filaments[:, 0]
    unique_ids, index_starts, counts = np.unique(
        ids, return_index=True, return_counts=True
    )

    # Compute end indices for each filament
    end_indices = index_starts + counts - 1

    # Initialize lists to store filaments for each pole
    filament_pole_lists = [[] for _ in range(poles.shape[0])]

    # Process each filament
    for idx, (start_idx, end_idx, filament_id) in enumerate(zip(index_starts, end_indices, unique_ids)):
        # Extract all points for the current filament
        filament_points = filaments[start_idx:end_idx + 1]
        filament_coords = filament_points[:, 1:4]

        # Calculate distances from all points to all poles
        differences = filament_coords[:, np.newaxis, :] - poles[np.newaxis, :, :]
        distances = np.linalg.norm(differences, axis=-1)  # Shape: [num_points, num_poles]

        # Find minimal distance to each pole
        min_distances = np.min(distances, axis=0)  # Shape: [num_poles]

        # Assign filament to the pole with the smallest minimal distance
        assigned_pole_index = np.argmin(min_distances)

        # Check distances of first and last points to the assigned pole
        dist_first = distances[0, assigned_pole_index]
        dist_last = distances[-1, assigned_pole_index]

        # Flip filament if the first point is farther than the last point
        if dist_first > dist_last:
            filament_points = filament_points[::-1]

        # Append to the appropriate pole's list
        filament_pole_lists[assigned_pole_index].append(filament_points)

    # Combine lists into arrays, handling empty cases
    filament_pole_arrays = [
        np.vstack(filament_list) if filament_list else np.empty((0, 4))
        for filament_list in filament_pole_lists
    ]

    return filament_pole_arrays
