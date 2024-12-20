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
from tardis_em.analysis.filament_utils import resample_filament, reorder_segments_id

from tardis_em.analysis.mt_classification.utils import (
    distances_of_ends_to_surface,
    count_true_groups,
    divide_into_sequences,
    fill_gaps,
    points_on_mesh_knn,
    select_mt_ids_within_bb,
    pick_pole_to_surfaces,
    assign_filaments_to_poles,
)
from tardis_em.utils.load_data import load_am_surf, ImportDataFromAmira
from tardis_em.utils.logo import TardisLogo


class MicrotubuleClassifier:
    """
    Manages the classification and preprocessing of data related to Microtubule (MT)
    analysis, including processing of surfaces, filaments, and poles. This class aims
    to facilitate segmentation and feature extraction for MT datasets, enabling further
    analysis tasks such as classification and structural evaluations.

    It initializes with critical parameters, loads and corrects processed data, and
    offers methods to extract relevant structural information from the provided files.
    """

    def __init__(
        self,
        surfaces: str,
        filaments: str,
        poles: str,
        pixel_size,
        gaps_size=100,
        kmt_dist_to_surf=1000,
        tardis_logo=True,
    ):
        """
        Initialize the class with parameters needed for data preprocessing and classification
        related to Microtubule (MT) analysis.

        :param surfaces: Path to the surface data file required for the analysis.
        :type surfaces: str
        :param filaments: Path to the filament data file required for the analysis.
        :type filaments: str
        :param poles: Path to the file containing poles information required for the analysis.
        :type poles: str
        :param pixel_size: Size of the pixels used in the data grid. Type is implied from usage.
        :type pixel_size: float or int
        :param gaps_size: The default size of gaps in the data grid (e.g., for separation); defaults
            to 100.
        :type gaps_size: int
        :param kmt_dist_to_surf: Distance of kinetochore microtubules from the surface grid
            in nanometers. Defaults to 1000.
        :type kmt_dist_to_surf: int
        :param tardis_logo: A boolean flag to toggle the display of the TARDIS logo and
            progress messages. Defaults to True.
        :type tardis_logo: bool
        """

        self.triangles = None
        self.tardis_logo = tardis_logo
        if self.tardis_logo:
            self.main_logo = TardisLogo()
            self.main_logo(
                title=f"| Transforms And Rapid Dimensionless Instance Segmentation",
                text_1="WELCOME to TARDIS_em - analysis [MT classification]!",
                text_2="Est. elapse time ~10 min",
            )

        self.pixel_size = pixel_size
        self.gap_size = gaps_size
        self.kmt_dist_to_surf = kmt_dist_to_surf
        self.surfaces = surfaces

        if self.tardis_logo:
            self.main_logo(
                title=f"| Transforms And Rapid Dimensionless Instance Segmentation",
                text_1="WELCOME to TARDIS_em - analysis [MT classification]!",
                text_2="Est. elapse time ~10 min",
                text_3="Classifying MT - Loading data...",
            )

        self.vertices = self.get_vertices_file(surfaces, simplify=None)
        self.filaments = self.get_filament_file(filaments)
        self.filament_pole1, self.filament_pole2 = None, None
        self.poles = self.get_poles_file(poles)
        self.min_coords = None

        if self.tardis_logo:
            self.main_logo(
                title=f"| Transforms And Rapid Dimensionless Instance Segmentation",
                text_1="WELCOME to TARDIS_em - analysis [MT classification]!",
                text_2="Est. elapse time ~10 min",
                text_3="Classifying MT - Starting preprocessing...",
            )
        self.correct_data()
        self.f_1 = len(np.unique(self.filament_pole1[:, 0]))
        self.f_2 = len(np.unique(self.filament_pole2[:, 0]))

        if self.tardis_logo:
            self.main_logo(
                title=f"| Transforms And Rapid Dimensionless Instance Segmentation",
                text_1="WELCOME to TARDIS_em - analysis [MT classification]!",
                text_2="Est. elapse time ~10 min",
                text_3="Classifying MT - Classifying...",
                text_4=f"Filament_pole_1: {self.f_1} | Filament_pole_1: {self.f_2 }",
                text_5="KMTs: NA",
                text_6="Mid-MT: NA",
                text_7="Interdigitating-MTs: NA",
                text_8="Bridging-MTs: NA",
                text_9=f"SMTs: NA",
            )

        self.plus_end_id, self.minus_end_id = None, None

        self.kmts_id_1, self.kmts_id_2 = None, None
        self.kmts_inside_id_1, self.kmts_outside_id_1 = None, None
        self.kmts_inside_id_2, self.kmts_outside_id_2 = None, None

        self.mid_mt_ids, self.int_mt_ids, self.brg_mt_ids = None, None, None
        self.smt_ids = None

    def get_vertices_file(self, dir_s: str, simplify: bool = None) -> list:
        """
        Extracts and returns the vertices information from a given directory containing
        data for an AM surface file. This method utilizes the `load_am_surf` function
        to process the data in the specified directory. Optionally, the simplification
        param can be used to streamline the vertices extraction process.

        :param dir_s: The directory containing the AM surface file.
        :type dir_s: str
        :param simplify: Defines whether simplification is applied during the loading
            process. This parameter is optional.
        :type simplify: bool, optional

        :return: A list of vertices extracted from the surface file.
        :rtype: list
        """
        _, _, vertices, self.triangles = load_am_surf(dir_s, simplify_f=simplify)

        return vertices

    @staticmethod
    def get_filament_file(dir_s: str) -> np.ndarray:
        """
        Extracts and returns segmented points from the specified directory using the
        ImportDataFromAmira class. This function is a static method and does not rely
        on class instance attributes.

        :param dir_s: Directory path to the source AM file
        :type dir_s: str

        :return: Numpy array of segmented points extracted from the specified directory
        :rtype: np.ndarray
        """
        return ImportDataFromAmira(src_am=dir_s).get_segmented_points()

    @staticmethod
    def get_poles_file(dir_s: str) -> np.ndarray:
        """
        Retrieve pole positions from a specified directory and returns them as a NumPy array.

        This method uses the `ImportDataFromAmira` class to access data associated with vertex
        positions in the specified directory. It processes the data to extract pole positions and
        return them in a structured format suitable for further computations or analysis.

        The directory passed must contain the required data structure that is compatible with
        `ImportDataFromAmira` functionality. The output is formatted as a NumPy array.

        :param dir_s: The file path to the directory containing the Amira data files
        :type dir_s: str

        :return: A NumPy array containing the vertex data representing pole positions
        :rtype: np.ndarray
        """
        return ImportDataFromAmira(src_am=dir_s).get_vertex()

    def correct_data(self):
        """
        Normalizes and processes filament data based on pixel size and assigned poles.

        This method performs several operations including:
        - Normalization of filament coordinates using the pixel size.
        - Resampling of the filament data.
        - Normalization of vertex and pole coordinates.
        - Assignment of filaments to poles and subsequent reordering.
        - Stacking of corrected filaments into a unified structure.
        """
        # Efficient normalization using broadcasting
        self.filaments[:, 1:] = self.filaments[:, 1:] / self.pixel_size
        self.filaments = resample_filament(self.filaments, 1)

        # Normalize vertices and poles
        for i in range(len(self.vertices)):
            self.vertices[i] = self.vertices[i] / self.pixel_size

        self.poles = (self.poles / self.pixel_size).astype(np.int32)
        self.poles = pick_pole_to_surfaces(self.poles, self.vertices)

        # Assign filaments to poles and reorder
        self.filament_pole1, self.filament_pole2 = assign_filaments_to_poles(
            self.filaments, self.poles
        )
        self.filament_pole1 = reorder_segments_id(self.filament_pole1)
        self.filament_pole2 = reorder_segments_id(self.filament_pole2)
        self.filament_pole2[:, 0] += np.max(self.filament_pole1[:, 0]) + 1

        # Stack the corrected filaments
        self.filaments = np.vstack((self.filament_pole1, self.filament_pole2))

    def get_filament_endpoints(self) -> tuple[int, int]:
        """
        Computes and returns the start and end indices of the unique filament IDs
        in the dataset. This function identifies unique filament IDs and uses them
        to derive the corresponding starting and ending positional indices.

        :return: A tuple containing two integers:
                 - The starting index of the first unique filament ID.
                 - The ending index of the last unique filament ID.
        :rtype: tuple[int, int]
        """
        ids = self.filaments[:, 0]
        unique_ids, index_starts, counts = np.unique(
            ids, return_index=True, return_counts=True
        )
        end_indices = int(index_starts + counts - 1)

        return int(index_starts), end_indices

    def assign_to_kmts(self, filaments, id_i=0) -> list:
        """
        Assigns filaments to kinetochores (KMTs) based on spatial and distance criteria.

        The function performs several steps to identify filaments, represented by their
        unique IDs, that meet specific conditions relative to bounding boxes, surfaces,
        and distance thresholds. It uses unique identification, filtering within bounding
        boxes, various distance measurements, and thresholds to determine the final set
        of KMT-associated filaments.

        :param filaments: A NumPy array representing the set of filaments.
        :param id_i: An integer representing the index of the vertices and poles
            being processed. Defaults to 0.
        :return: A list of unique filament IDs assigned to KMTs.
        """
        _, unique_indices = np.unique(filaments[:, 0], return_index=True)
        plus_end = filaments[unique_indices, :]

        # Preselect filaments inside bounding box
        kmt_ids = select_mt_ids_within_bb(self.vertices[id_i], plus_end)
        if len(kmt_ids) == 0:
            return []

        kmt_fibers = filaments[np.isin(filaments[:, 0], kmt_ids)]
        kmt_ids = self.assign_mt_with_crossing(kmt_fibers, self.vertices[id_i], [1])

        if len(kmt_ids) == 0:
            return []

        # Calculate distances of MT endpoints to the surface and poles
        d1_, d2_ = distances_of_ends_to_surface(
            self.vertices[id_i], self.poles[id_i], plus_end
        )

        # Select MTs based on distance threshold
        dist_to_surf_th = d1_ <= (self.kmt_dist_to_surf / self.pixel_size)
        d1_ = d1_[dist_to_surf_th]
        d2_ = d2_[dist_to_surf_th]
        plus_end = plus_end[dist_to_surf_th[:, 0]]

        # Select MTs where the pole distance is greater than the surface distance
        plus_end = np.unique(plus_end[d2_ > d1_, 0])

        # Combine and return final KMT IDs
        return list(np.unique(np.hstack((kmt_ids, plus_end))))

    def kmts_inside_outside(
        self, kmt_proposal, id_i=0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Determines whether the k-MTs (kinetochore MicroTubules) are inside or outside based
        on their distance to the surface. The function computes unique k-MTs and evaluates
        the proximity of their plus ends in comparison to a given pole surface. The result
        is segregated into k-MTs considered inside and outside.

        :param kmt_proposal: A 2D numpy array where each row corresponds to an association
            of k-MTs, with the first column representing k-MT identifiers.
        :param id_i: An integer representing the identifier of the current vertex and pole
            configuration to be used for computations. Defaults to 0.
        :return: A tuple containing two numpy arrays. The first array contains unique k-MT
            identifiers classified as inside, while the second array contains unique k-MT
            identifiers classified as outside.
        """
        _, unique_indices = np.unique(kmt_proposal[:, 0], return_index=True)
        plus_ends = kmt_proposal[unique_indices, :]

        d1_, d2_ = distances_of_ends_to_surface(
            self.vertices[id_i], self.poles[id_i], plus_ends, True
        )
        d1_to_d2 = d2_ > d1_

        kmt_ids_inside = np.hstack(
            [
                kmt_proposal[kmt_proposal[:, 0] == i, 0]
                for i in np.unique(kmt_proposal[:, 0])[d1_to_d2[:, 0]]
            ]
        )
        kmt_ids_outside = np.hstack(
            [
                kmt_proposal[kmt_proposal[:, 0] == i, 0]
                for i in np.unique(kmt_proposal[:, 0])[~d1_to_d2[:, 0]]
            ]
        )

        return np.unique(kmt_ids_inside), np.unique(kmt_ids_outside)

    def assign_to_mid_mt(self) -> list:
        """
        Assigns microtubules (MTs) to a middle set based on specific conditions and filters.

        The function identifies MTs which are not part of specific pre-defined ID sets and
        selects candidate MTs with no more than one crossing. It further filters these selected
        MTs based on their distance to a defined set of vertices in comparison to poles. The
        final IDs of the specific MTs that meet all conditions are returned as a list.

        :return: A list of MT IDs that belong to the middle set. These are filtered based on
            their geometric properties and crossing conditions.
        :rtype: list
        """
        mts_plus = np.vstack(
            [
                self.filaments[self.filaments[:, 0] == i, :][0]
                for i in np.unique(self.filaments[:, 0])
                if i not in list(np.concatenate((self.kmts_id_1, self.kmts_id_2)))
            ]
        )
        mts_minus = np.vstack(
            [
                self.filaments[self.filaments[:, 0] == i, :][-1]
                for i in np.unique(self.filaments[:, 0])
                if i not in list(np.concatenate((self.kmts_id_1, self.kmts_id_2)))
            ]
        )

        """Select MT with 0 or 1 crossing"""
        mid_mt_ids = select_mt_ids_within_bb(
            vertices_n=np.vstack(self.vertices), mt_ends1=mts_plus, mt_ends2=mts_minus
        )
        if len(mid_mt_ids) == 0:
            return []

        mid_mt_fibers = self.filaments[np.isin(self.filaments[:, 0], mid_mt_ids)]
        mid_mt_ids = self.assign_mt_with_crossing(
            filaments=mid_mt_fibers, vertices_l=np.vstack(self.vertices), class_l=[0, 1]
        )
        if len(mid_mt_ids) == 0:
            return []

        """Filter out MT that have (-) end closer to the center of vertices then pole"""
        knn_poles = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
            np.vstack((self.poles, np.mean(np.vstack(self.vertices), axis=0)))
        )

        _, indices = knn_poles.kneighbors(
            np.vstack([mts_minus[mts_minus[:, 0] == i, 1:] for i in mid_mt_ids])
        )
        indices = (indices == 2).flatten()
        mid_mt_ids = mid_mt_ids[indices]

        return list(mid_mt_ids)

    def assign_to_int_mt(self) -> list:
        """
        Assigns intermediate microtubule IDs based on specific criteria.

        This function identifies intermediate microtubules (MTs) from given input data,
        excluding specific MT IDs and using bounding boxes to further select relevant
        MTs. It then reassigns the identified MTs based on crossing criteria and returns
        their IDs as a list.

        :raises ValueError: If the provided data structures have inconsistent shapes or types.

        :return: A list of assigned intermediate microtubule IDs. If no IDs are found or
                 assigned, an empty list is returned.
        :rtype: list
        """
        mts_plus = np.vstack(
            [
                self.filaments[self.filaments[:, 0] == i, :][0]
                for i in np.unique(self.filaments[:, 0])
                if i
                not in list(
                    np.concatenate((self.kmts_id_1, self.kmts_id_2, self.mid_mt_ids))
                )
            ]
        )

        int_mt_ids = select_mt_ids_within_bb(np.vstack(self.vertices), mts_plus)

        if len(int_mt_ids) == 0:
            return []

        int_mt_fibers = self.filaments[np.isin(self.filaments[:, 0], int_mt_ids)]
        int_mt_ids = self.assign_mt_with_crossing(
            filaments=int_mt_fibers, vertices_l=np.vstack(self.vertices), class_l=[2, 3]
        )

        return list(int_mt_ids)

    def assign_to_bridge_mt(self) -> list:
        """
        Assigns microtubules to the bridge class based on their ID exclusions and their spatial
        crossings with assigned vertices. It first filters out specific filament IDs that belong
        to other microtubule classes and then assigns remaining filaments to the bridge if they
        meet crossing conditions.

        :returns: A list of microtubule IDs assigned to the bridge class.
        :rtype: list
        """
        brg_mt_fibers = self.filaments[
            ~np.isin(
                self.filaments[:, 0],
                list(
                    np.concatenate(
                        (
                            self.kmts_id_1,
                            self.kmts_id_2,
                            self.mid_mt_ids,
                            self.int_mt_ids,
                        )
                    )
                ),
            )
        ]

        bridge_mt_ids = self.assign_mt_with_crossing(
            filaments=brg_mt_fibers,
            vertices_l=np.vstack(self.vertices),
            class_l=[4, 5, 6],
        )
        return list(bridge_mt_ids)

    def assign_mt_with_crossing(self, filaments, vertices_l, class_l=[1]) -> np.ndarray:
        """
        Assigns microtubule (MT) IDs based on whether or not they cross a specific surface
        and belong to a specified class of interest.

        The function evaluates a set of segmented microtubules (filaments) and determines
        to which class each should belong. This classification is based on their crossing
        behavior with a given surface defined by vertices, as well as a specified class
        parameter. It ensures that MTs crossing specific surfaces or having other
        characteristics are correctly classified.

        :param filaments: A 2D numpy array where each row represents a segment of a
            microtubule. The first column contains microtubule IDs (integers), while
            the subsequent columns correspond to the spatial coordinates (e.g., x, y, z).
        :param vertices_l: A 2D numpy array representing the vertices of the surface for
            comparison, each row specifying a spatial coordinate (e.g., x, y, z).
        :param class_l: A list of integers representing the microtubule classification
            criteria. Default is [1], which selects microtubules matching the threshold
            condition.
        :return: A numpy array containing IDs of microtubules that intersect the surface
            and fit the classification criteria specified by the `class_` parameter.
        """
        all_ids = np.unique(filaments[:, 0])

        """Calculate MT crossing the surface"""
        _, points_on_surface = points_on_mesh_knn(filaments[:, 1:], vertices_l)

        points_indices = [id_ for id_, i in enumerate(points_on_surface) if i]
        mt_id_crossing = np.unique(filaments[points_indices, 0]).astype(np.int16)

        point_sequence = fill_gaps(points_indices, self.gap_size)
        point_sequence = [
            item
            for sublist in divide_into_sequences(np.unique(point_sequence))
            for item in sublist
        ]
        point_sequence = np.array(
            [
                True if i in point_sequence else False
                for i in list(range(0, len(filaments)))
            ]
        )

        """Select MT with one crossing"""
        ids = []
        for mt_id in mt_id_crossing:
            f = filaments[:, 0] == mt_id
            f = point_sequence[f,]

            if count_true_groups(f) in class_l:
                ids.append(mt_id)

        if 0 in class_l:
            for mt_id in all_ids:
                if mt_id not in mt_id_crossing:
                    ids.append(mt_id)
        return np.array(ids)

    def classified_MTs(self):
        """
        Classifies microtubules (MTs) into different categories based on spatial properties, assignments to poles,
        and other criteria. The classification includes Kinetochore Microtubules (KMTs), Mid-MTs, Interdigitating-MTs,
        Bridging-MTs, and Standard Microtubules (SMTs). Coordinates are corrected to the original scale after assignments.
        Contains options for a logo display summarizing the classification process. This function integrates multiple
        steps such as endpoint determination, pole classifications, and other custom classifications.

        :param self: The object instance invoking the method.

        :raises ValueError: If any invalid indices are encountered during processing.
        :raises TypeError: If inputs to certain classification methods have mismatched types.
        :raises AttributeError: If required attributes are missing from the self instance.

        :return: None
        """
        """Get indices for ends"""
        self.plus_end_id, self.minus_end_id = self.get_filament_endpoints()

        """Get indices for KMTs"""
        self.kmts_id_1 = list(
            self.assign_to_kmts(filaments=self.filament_pole1, id_i=0)
        )
        self.kmts_id_2 = list(
            self.assign_to_kmts(filaments=self.filament_pole2, id_i=1)
        )

        self.kmts_inside_id_1, self.kmts_outside_id_1 = self.kmts_inside_outside(
            self.filaments[np.isin(self.filaments[:, 0], self.kmts_id_1)], id_i=0
        )
        self.kmts_inside_id_1, self.kmts_outside_id_1 = list(
            self.kmts_inside_id_1
        ), list(self.kmts_outside_id_1)
        self.kmts_outside_id_2, self.kmts_inside_id_2 = self.kmts_inside_outside(
            self.filaments[np.isin(self.filaments[:, 0], self.kmts_id_2)], id_i=1
        )
        self.kmts_outside_id_2, self.kmts_inside_id_2 = list(
            self.kmts_outside_id_2
        ), list(self.kmts_inside_id_2)

        if self.tardis_logo:
            self.main_logo(
                title=f"| Transforms And Rapid Dimensionless Instance Segmentation",
                text_1="WELCOME to TARDIS_em - analysis [MT classification]!",
                text_2="Est. elapse time ~10 min",
                text_3="Classifying MT - Classifying...",
                text_4=f"Filament_pole_1: {self.f_1} | Filament_pole_1: {self.f_2 }",
                text_5=f"KMTs: {len(self.kmts_id_1)+len(self.kmts_id_2)}",
                text_6="Mid-MT: NA",
                text_7="Interdigitating-MTs: NA",
                text_8="Bridging-MTs: NA",
                text_9=f"SMTs: NA",
            )

        # Select and assign mid-MTs, interdigitating-MTs, and bridging-MTs
        self.mid_mt_ids = self.assign_to_mid_mt()
        if self.tardis_logo:
            self.main_logo(
                title=f"| Transforms And Rapid Dimensionless Instance Segmentation",
                text_1="WELCOME to TARDIS_em - analysis [MT classification]!",
                text_2="Est. elapse time ~10 min",
                text_3="Classifying MT - Classifying...",
                text_4=f"Filament_pole_1: {self.f_1} | Filament_pole_1: {self.f_2 }",
                text_5=f"KMTs: {len(self.kmts_id_1)+len(self.kmts_id_2)}",
                text_6=f"Mid-MT: {len(self.mid_mt_ids)}",
                text_7="Interdigitating-MTs: NA",
                text_8="Bridging-MTs: NA",
                text_9=f"SMTs: NA",
            )

        self.int_mt_ids = self.assign_to_int_mt()
        if self.tardis_logo:
            self.main_logo(
                title=f"| Transforms And Rapid Dimensionless Instance Segmentation",
                text_1="WELCOME to TARDIS_em - analysis [MT classification]!",
                text_2="Est. elapse time ~10 min",
                text_3="Classifying MT - Classifying...",
                text_4=f"Filament_pole_1: {self.f_1} | Filament_pole_1: {self.f_2 }",
                text_5=f"KMTs: {len(self.kmts_id_1)+len(self.kmts_id_2)}",
                text_6=f"Mid-MT: {len(self.mid_mt_ids)}",
                text_7=f"Interdigitating-MTs: {len(self.int_mt_ids)}",
                text_8="Bridging-MTs: NA",
                text_9=f"SMTs: NA",
            )

        self.brg_mt_ids = self.assign_to_bridge_mt()

        # Correct coordinates back to their original state
        self.filaments[:, 1:] = self.filaments[:, 1:] * self.pixel_size
        self.poles = self.poles * self.pixel_size

        for i in range(len(self.vertices)):
            self.vertices[i] = self.vertices[i] * self.pixel_size

        """Select SMTs"""
        self.smt_ids = (
            self.kmts_id_1
            + self.kmts_id_2
            + self.mid_mt_ids
            + self.int_mt_ids
            + self.brg_mt_ids
        )
        self.smt_fiber = self.filaments[~np.isin(self.filaments[:, 0], self.smt_ids)]

        self.smt_ids = list(np.unique(self.smt_fiber[:, 0]))

        if self.tardis_logo:
            self.main_logo(
                title=f"| Transforms And Rapid Dimensionless Instance Segmentation",
                text_1="WELCOME to TARDIS_em - analysis [MT classification]!",
                text_2="Est. elapse time ~10 min",
                text_3="Classifying MT - Classifying...",
                text_4=f"Filament_pole_1: {self.f_1} | Filament_pole_1: {self.f_2}",
                text_5=f"KMTs: {len(self.kmts_id_1) + len(self.kmts_id_2)}",
                text_6=f"Mid-MT: {len(self.mid_mt_ids)}",
                text_7=f"Interdigitating-MTs: {len(self.int_mt_ids)}",
                text_8=f"Bridging-MTs: {len(self.brg_mt_ids)}",
                text_9=f"SMTs: {len(self.smt_ids)}",
            )

    def get_classified_indices(self) -> list:
        """
        Returns a list of classified indices, grouped into specific categories
        representing various groups of IDs. These categories include inside and
        outside IDs, mid-point IDs, intermediate IDs, bridge IDs, and SMT IDs.

        :return: List containing groups of classified indices.
        :rtype: list
        """
        return [
            [self.kmts_inside_id_1, self.kmts_outside_id_1],
            [self.kmts_inside_id_2, self.kmts_outside_id_2],
            self.mid_mt_ids,
            self.int_mt_ids,
            self.brg_mt_ids,
            self.smt_ids,
        ]

    def get_classified_fibers(self) -> list:
        """
        Classifies and retrieves fibers based on their categories and identifiers. This method
        divides the fibers into multiple subcategories including kinetochore microtubules
        (kMT) for two different states (inside and outside), midplane microtubules
        (mid_MT), interpolar microtubules (int_MT), bridging microtubules (brg_MT),
        and other spindle microtubules (smt_MT). The classification is performed
        by checking the fiber identifiers against predefined ID collections.

        :return: A nested list where each element represents a specific category or subcategory
                 of classified fibers. It includes a list of kMT fibers for each state
                 (inside, outside), and other categorized fibers (mid_MT, int_MT, brg_MT, smt_MT).
        :rtype: list
        """
        kmt_fiber_inside_1 = self.filaments[
            np.isin(self.filaments[:, 0], self.kmts_inside_id_1)
        ]
        kmt_fiber_outside_1 = self.filaments[
            np.isin(self.filaments[:, 0], self.kmts_outside_id_1)
        ]
        kmt_fiber_inside_2 = self.filaments[
            np.isin(self.filaments[:, 0], self.kmts_inside_id_2)
        ]
        kmt_fiber_outside_2 = self.filaments[
            np.isin(self.filaments[:, 0], self.kmts_outside_id_2)
        ]

        mid_mt_fiber = self.filaments[np.isin(self.filaments[:, 0], self.mid_mt_ids)]
        int_mt_fiber = self.filaments[np.isin(self.filaments[:, 0], self.int_mt_ids)]
        brg_mt_fiber = self.filaments[np.isin(self.filaments[:, 0], self.brg_mt_ids)]

        return [
            [kmt_fiber_inside_1, kmt_fiber_outside_1],
            [kmt_fiber_inside_2, kmt_fiber_outside_2],
            mid_mt_fiber,
            int_mt_fiber,
            brg_mt_fiber,
            self.smt_fiber,
        ]

    def get_filaments(self) -> np.ndarray:
        """
        Returns the filaments stored in the object.

        The function provides direct access to the filaments as a
        numpy array. It is useful for retrieving the stored filaments
        data for further processing or analysis purposes.

        :return: A numpy array containing filaments data.
        :rtype: np.ndarray
        """
        return self.filaments

    def get_vertices(self, simplify=128) -> tuple[list, list]:
        """
        Retrieve the vertices and triangles of the object, with optional simplification.

        The method allows fetching the vertices and triangles that represent the geometry
        of the object. By default, an optional simplification value can be provided to
        reduce the complexity of the vertices. If the `simplify` parameter is `None`,
        an alternative retrieval process is utilized.

        :param simplify: Specifies the simplification factor for reducing vertex complexity.
                         If `None`, returns the detailed geometry without simplification.
        :type simplify: int or None
        :return: A tuple containing a list of vertices and a list of triangles.
        :rtype: tuple[list, list]
        """
        if simplify is not None:
            return self.vertices, self.triangles
        else:
            _, _, vertices, self.triangles = self.get_vertices_file(
                self.surfaces, simplify
            )
            return vertices, self.triangles

    def get_poles(self) -> np.ndarray:
        """
        Retrieves the poles data of a given instance and returns it as a NumPy array.
        Poles typically represent key computation parameters or specific physical
        characteristics depending on the context of implementation.

        :return: An array containing the poles of the current instance.

        :rtype: numpy.ndarray
        """

        return self.poles
