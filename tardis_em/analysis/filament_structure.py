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


def assign_filaments_to_poles(filaments, poles):
    """
    Assigns filaments to the closest poles and potentially flips their orientation to maintain consistency
    based on proximity to the assigned pole.

    The function processes a set of filaments, identified by unique IDs, and calculates their start and end points.
    Distances from these points to given poles are computed, and each filament is assigned to the nearest pole.
    If needed, the points of a filament are reordered to ensure that their orientation starts closer to their assigned pole.
    Filaments are then grouped according to their assigned pole.

    :param filaments: A 2D NumPy array where each row contains information about an individual
        point of a filament. The first column represents the filament ID, and the
        remaining columns represent the position coordinates of the point.
    :param poles: A 2D NumPy array where each row contains the coordinates of a pole.
    :return: A tuple containing two arrays. The first array represents the filaments assigned
        to the first pole, while the second array represents the filaments assigned to the
        second pole. Each filament is listed with its points in correct order to maintain
        proximity to the assigned pole.
    :rtype: Tuple[np.ndarray, np.ndarray]
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
        if dist_start > dist_end:
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
