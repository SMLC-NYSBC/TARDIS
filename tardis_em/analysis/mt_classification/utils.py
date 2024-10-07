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


def count_true_groups(bool_list):
    count = 0
    in_group = False

    for value in bool_list:
        if value and not in_group:
            count += 1
            in_group = True
        elif not value:
            in_group = False
    return count


def distances_of_ends_to_surface(vertices_, pole_, ends, d1_to_surf=False):
    knn_v = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(vertices_)
    knn_e = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(pole_.reshape(1, -1))

    if d1_to_surf:
        _, i1 = knn_v.kneighbors(ends[:, 1:])
        d1, _ = knn_e.kneighbors(vertices_[i1.flatten(), :])
    else:
        d1, _ = knn_v.kneighbors(ends[:, 1:])
    d2, _ = knn_e.kneighbors(ends[:, 1:])

    return d1, d2


def divide_into_sequences(arr):
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


def fill_gaps(float_list, n):
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


def pick_pole_to_surfaces(poles, vertices):
    v_centroid = np.mean(vertices[0], axis=0)

    p1_centroid = np.linalg.norm(v_centroid - poles[0, :])
    p2_centroid = np.linalg.norm(v_centroid - poles[1, :])

    if p1_centroid > p2_centroid:
        return np.array([poles[1, :], poles[0, :]])
    else:
        return poles


def points_on_mesh_knn(points, vertices):
    knn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(vertices)
    distances, _ = knn.kneighbors(vertices)
    mean_distance = np.mean(distances[:, 1])

    distances, _ = knn.kneighbors(points)
    distances_bool = distances[:, 0] <= mean_distance

    return distances[:, 0], np.array(distances_bool)


def select_mt_ids_within_bb(vertices_, mt_ends):
    # Vertices 3D bounding box
    min_x, max_x = np.min(vertices_[:, 0]), np.max(vertices_[:, 0])
    min_y, max_y = np.min(vertices_[:, 1]), np.max(vertices_[:, 1])
    min_z, max_z = np.min(vertices_[:, 2]), np.max(vertices_[:, 2])

    def filter_mts(min_, max_, axis):
        mask = (mt_ends[:, axis] >= min_) & (mt_ends[:, axis] <= max_)
        return mt_ends[mask]

    mt_ends = filter_mts(min_x, max_x, axis=1)
    mt_ends = filter_mts(min_y, max_y, axis=2)
    mt_ends = filter_mts(min_z, max_z, axis=3)

    return np.unique(mt_ends[:, 0])
