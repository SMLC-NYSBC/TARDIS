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

from analysis.mt_classification.utils import (
    distances_of_ends_to_surface,
    count_true_groups,
    divide_into_sequences,
    fill_gaps,
    points_on_mesh_knn,
    select_mt_ids_within_bb,
)


def assign_filaments_to_poles(filaments, poles):
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


def assign_filaments_to_ends(filaments):
    """
    Get plus/minus ends indices

    Args:
        - filaments (np.ndarray):

    Returns:
        - array of indices for plus and minus ends
    """
    _, unique_indices = np.unique(filaments[:, 0], return_index=True)
    plus_ends = unique_indices.T

    minus_ends = np.zeros_like(plus_ends)
    minus_ends[:-1] = (plus_ends - 1)[1:]
    minus_ends[-1] = len(filaments) - 1

    return plus_ends, minus_ends


def assign_filaments_to_kmts(filaments, vertices, pole, kmt_dist_to_surf=1000, pixel_size=25.7):
    """Select MT within surface BB"""
    _, unique_indices = np.unique(filaments[:, 0], return_index=True)
    plus_end = filaments[unique_indices, :]

    kmt_ids = select_mt_ids_within_bb(vertices, plus_end)
    kmt_fibers = np.vstack([filaments[filaments[:, 0] == i, :] for i in kmt_ids])

    """Calculate MT crossing the surface"""
    _, points_on_surface = points_on_mesh_knn(kmt_fibers[:, 1:], vertices)

    points_indices = [id_ for id_, i in enumerate(points_on_surface) if i]
    kmt_id_crossing = np.unique(kmt_fibers[points_indices, 0]).astype(np.int16)

    point_sequence = fill_gaps(points_indices, 15)
    point_sequence = [
        item
        for sublist in divide_into_sequences(np.unique(point_sequence))
        for item in sublist
    ]
    point_sequence = np.array(
        [
            True if i in point_sequence else False
            for i in list(range(0, len(kmt_fibers)))
        ]
    )

    """Select MT with one crossing"""
    kmt_ids = []
    for kmt_id in kmt_id_crossing:
        f = kmt_fibers[:, 0] == kmt_id
        f = point_sequence[f,]

        if count_true_groups(f) == 1:
            kmt_ids.append(kmt_id)
    kmt_ids = np.array(kmt_ids)

    """Compute (+)-end distance of MT to the NN vertices (d1) and the pole (d2)"""
    d1_, d2_ = distances_of_ends_to_surface(vertices, pole, plus_end)

    """Select MT which d1 <= distance threshold"""
    dist_to_surf_th = d1_ <= (kmt_dist_to_surf / pixel_size)
    d1_ = d1_[dist_to_surf_th]
    d2_ = d2_[dist_to_surf_th]
    plus_end = plus_end[dist_to_surf_th[:, 0]]

    """Select MT which d1 > d2"""
    d1_to_d2 = d2_ > d1_
    plus_end = np.unique(plus_end[d1_to_d2, 0])

    """Combine both KMTs"""
    kmt_ids = np.unique(np.hstack((kmt_ids, plus_end)))

    return kmt_ids


def assign_kmt_inside_outside(kmt_filaments, vertices, pole):
    _, unique_indices = np.unique(kmt_filaments[:, 0], return_index=True)
    plus_ends = kmt_filaments[unique_indices, :]

    d1_, d2_ = distances_of_ends_to_surface(vertices, pole, plus_ends, True)
    d1_to_d2 = d2_ > d1_

    kmt_ids_inside = np.hstack(
        [
            kmt_filaments[kmt_filaments[:, 0] == i, 0]
            for i in np.unique(kmt_filaments[:, 0])[d1_to_d2[:, 0]]
        ]
    )
    kmt_ids_outside = np.hstack(
        [
            kmt_filaments[kmt_filaments[:, 0] == i, 0]
            for i in np.unique(kmt_filaments[:, 0])[~d1_to_d2[:, 0]]
        ]
    )

    return np.unique(kmt_ids_inside), np.unique(kmt_ids_outside)


