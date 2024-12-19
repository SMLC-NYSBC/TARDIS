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
    Counts the number of consecutive groups of `True` values in a list.

    Args:
        bool_list (list): A list of boolean values where groups of `True` values are to be counted.

    Return:
        int: The total number of groups of `True` values in the input list.
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
    vertices_, pole_, ends, d1_to_surf=False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the distances of the specified endpoints to the nearest points on the
    surface represented by the given vertices and pole. The function offers flexibility
    via the `d1_to_surf` parameter to include vertex-referenced nearest neighbors
    calculation if required.

    Args:
        vertices_ (np.ndarray): An array of points defining the surface.
        pole_ (np.ndarray): A single point coordinates for distance calculations.
        ends (np.ndarray): An array of endpoints for which the distance is to be calculated.
        d1_to_surf (bool): A boolean flag that, when set to True, calculates the nearest neighbor distances
            for the first column of `ends` specifically to `vertices_` and then finds the distance to `pole_`.

    Returns:
        tuple: A tuple containing two arrays:
            - Distances of the endpoints to the vertices or adjusted surface.
            - Distances of the endpoints to the pole.
    """
    knn_v = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(vertices_)
    knn_e = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(pole_.reshape(1, -1))

    if d1_to_surf:
        _, i1 = knn_v.kneighbors(ends[:, 1:])
        d1, _ = knn_e.kneighbors(vertices_[i1.flatten(), :])
    else:
        d1, _ = knn_v.kneighbors(ends[:, 1:])
    d2, _ = knn_e.kneighbors(ends[:, 1:])

    return d1, d2


def distance_to_the_pole(points: np.ndarray, distance_to: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean distance of given points from a reference point.

    Args:
        points (np.ndarray): Array of points for which the distance needs to be calculated.
        distance_to (np.ndarray): Array representing the reference point.

    Return:
         A numpy array of distances for each point in the input array..
    """
    distances = np.sqrt(np.sum((points - distance_to) ** 2, axis=1))

    return distances


def divide_into_sequences(arr) -> list[list[int]]:
    """
    Divides a list of integers into sequences of consecutive numbers.

    Args:
        arr (list): The input list of integers to be divided into sequences.
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
    Fills the gaps in a list of floating-point numbers by iteratively inserting
    intermediate values between consecutive elements if the difference between
    them exceeds a given threshold `n`.

    The filled list is then returned with unique values in ascending order.

    Args:
        float_list (list): A list of floating-point numbers where gaps need to be filled.
        n (float): A float representing the threshold for the maximum allowable
              difference between consecutive elements in the list.
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
    Computes and selects the order of poles relative to the centroid of the
    first set of vertices by comparing the distances of each pole to the
    centroid. Returns the poles in the order determined by their proximity
    to the centroid.

    Args:
        poles (np.ndarray): 2D numpy array of shape (2, n) representing two poles.
        vertices (np.ndarray): 3D numpy array with shape (m, k, n), where m represents
            the number of vertex groups, k the number of vertices per group,
            and n the dimensions of vertices. The array represents the set of
            vertices organized in groups, with this function using only the
            first group of vertices for computation.
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
    Calculate distances from given points to the nearest vertices on a mesh and
    determine whether these distances fall within a defined threshold of the
    mean inter-vertex distance of the mesh.

    Args:
        points (np.ndarray): An array of 3D coordinates representing the points to be checked against the mesh.
        vertices (np.ndarray): An array of 3D coordinates representing the vertices of the mesh.
    """
    knn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(vertices)
    distances, _ = knn.kneighbors(vertices)
    mean_distance = np.mean(distances[:, 1]) * 2

    distances, _ = knn.kneighbors(points)
    distances_bool = distances[:, 0] <= mean_distance

    return distances[:, 0], np.array(distances_bool)


def select_mt_ids_within_bb(
    vertices_: np.ndarray, mt_ends1: np.ndarray, mt_ends2=None
) -> np.ndarray:
    """
    Selects microtubule (MT) IDs whose endpoints fall within the 3D bounding box defined
    by the provided vertices. The function checks if the MT endpoints in ``mt_ends1`` and
    potentially ``mt_ends2`` lie within the 3D range defined by the minimum and maximum
    coordinates of the vertices. If ``mt_ends2`` is provided, cross-referencing is performed
    to refine the list of MT IDs.

    Args:
        vertices_ (np.ndarray): 3D vertices array of shape (N, 3), where N is the number of vertices.
            Each row represents the (x, y, z) coordinates of a vertex.
        mt_ends1 (np.ndarray): Array containing microtubule endpoint data of shape (M, 4), where M is
            the number of endpoints. Each row represents one endpoint in the format
            [MT_ID, x, y, z], where MT_ID is the identifier of the microtubule.
        mt_ends2 (np.ndarray): Optional array similar to ``mt_ends1`` containing additional
            microtubule endpoint data to be cross-referenced with the filtered MT IDs from
            ``mt_ends1``. If provided, it undergoes additional filtering to refine results.

    Returns:
        np.ndarray: Array of unique MT IDs whose endpoints fall within the 3D bounding box defined by
        the provided vertices. If ``mt_ends2`` is provided, the IDs are further filtered using
        cross-referencing between ``mt_ends1`` and ``mt_ends2``.
    """
    # Vertices 3D bounding box
    min_x, max_x = np.min(vertices_[:, 0]), np.max(vertices_[:, 0])
    min_y, max_y = np.min(vertices_[:, 1]), np.max(vertices_[:, 1])
    min_z, max_z = np.min(vertices_[:, 2]), np.max(vertices_[:, 2])

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
    Assign filaments to the nearest pole based on the minimal distance from filament endpoints to poles,
    and flip filaments (if needed) so that the end closest to the assigned pole is always at the bottom.

    Args:
        filaments (np.ndarray): Array of shape (n, 4) with columns [ID, X, Y, Z].
        poles (np.ndarray): Array of shape (2, 3) with coordinates of the two poles.

    Returns:
        filament_pole1, filament_pole2 (np.ndarray): Array containing filaments assigned to Pole 1, 2.
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
