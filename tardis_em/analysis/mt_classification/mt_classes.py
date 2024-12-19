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
    MicrotubuleClassifier is responsible for analyzing and classifying microtubule (MT)
    filaments using input data related to spatial graph files, surfaces, and poles.

    This class performs various preprocessing steps such as loading and correcting
    data, classifying microtubules into different categories like Kinetochore
    Microtubules (KMTs), bridging microtubules, and spindle microtubules, among others.
    Additionally, it visualizes progress with optional logging via the TardisLogo feature.

    :ivar triangles: Stores triangle"""

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
        Initializes the Microtubule Classifier with the necessary data.

        Args:
            filaments (str): File directory with amira spatial graph.
            surfaces (str): File directory with amira surface.
            poles (str): File directory with amira spatial graph for poles
            pixel_size (float): Pixel size value.
            gaps_size (int): Gap size to be fill out in case of detected
                uneven surface crossing.
            kmt_dist_to_surf (int): Distance of plus end MT to the surface in A
                to be considered as KMT.
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

    def get_vertices_file(self, dir_: str, simplify: bool = None) -> list:
        """
        Retrieve the vertices from a file within a specified directory. This function
        is responsible for invoking the `load_am_surf` function to load vertex data
        from a given directory and optionally simplifying it, based on the supplied
        parameters.

        Args:
            dir_ (str): The directory containing the file used to load the surface data.
            simplify (bool): A flag indicating whether to simplify the retrieved vertex data.
        """
        _, _, vertices, self.triangles = load_am_surf(dir_, simplify_=simplify)

        return vertices

    @staticmethod
    def get_filament_file(dir_) -> np.ndarray:
        """
        Retrieves and returns the filament file data based on the specified directory.
        """
        return ImportDataFromAmira(src_am=dir_).get_segmented_points()

    @staticmethod
    def get_poles_file(dir_) -> np.ndarray:
        """
        This static method retrieves the poles file from a specified directory.
        """
        return ImportDataFromAmira(src_am=dir_).get_vertex()

    def correct_data(self):
        """
        Efficiently normalizes filament data and reorganizes pole assignment for a more structured setup.
        The normalized and corrected filament data are consolidated into a single array.
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
        Returns the start and end points of each filament.

        This function identifies the start and end indices for each filament
        based on their unique identifiers.
        """
        ids = self.filaments[:, 0]
        unique_ids, index_starts, counts = np.unique(
            ids, return_index=True, return_counts=True
        )
        end_indices = int(index_starts + counts - 1)

        return int(index_starts), end_indices

    def assign_to_kmts(self, filaments, id_=0) -> list:
        """
        Assign filaments to kinetochore microtubules (KMTs) based on their distances,
        surface crossing criteria, and proximity to cellular structures.

        This function identifies microtubules (MTs) that belong to KMTs by analyzing
        geometrical properties and relationships between MT endpoints, surface vertices,
        and poles of cellular structures.

        Args:
            filaments: An array containing filament data.
            id_: The identifier index for selecting the cellular structure
            vertices and poles from the corresponding data.
        """
        _, unique_indices = np.unique(filaments[:, 0], return_index=True)
        plus_end = filaments[unique_indices, :]

        # Preselect filaments inside bounding box
        kmt_ids = select_mt_ids_within_bb(self.vertices[id_], plus_end)
        if len(kmt_ids) == 0:
            return []

        kmt_fibers = filaments[np.isin(filaments[:, 0], kmt_ids)]
        kmt_ids = self.assign_mt_with_crossing(kmt_fibers, self.vertices[id_], [1])

        if len(kmt_ids) == 0:
            return []

        # Calculate distances of MT endpoints to the surface and poles
        d1_, d2_ = distances_of_ends_to_surface(
            self.vertices[id_], self.poles[id_], plus_end
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

    def kmts_inside_outside(self, kmt_proposal, id_=0) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates whether given kinetochore-microtubule (KMT) proposals are located
        inside or outside the defined surface. The classification is based on the
        distances between ends of the KMT proposals and the specified surface, where
        each proposal corresponds to unique kinetochore identifiers.

        This method aims to provide classified identifiers of KMTs that are either
        inside or outside, enabling further analysis or tracking of microtubule
        dynamics and interactions with defined structures.

        Args:
            kmt_proposal (np.ndarray): Array containing KMT proposals.
            id_ (int): Optional identifier, defaulting to 0, which specifies the surface
                    or context against which distances are calculated.

        Return:
            - First array contains unique identifiers of KMTs classified as inside.
            - Second array contains unique identifiers of KMTs classified as outside.
        """
        _, unique_indices = np.unique(kmt_proposal[:, 0], return_index=True)
        plus_ends = kmt_proposal[unique_indices, :]

        d1_, d2_ = distances_of_ends_to_surface(
            self.vertices[id_], self.poles[id_], plus_ends, True
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
        Assigns microtubules (MTs) to the middle region based on specific geometric and
        structural conditions.

        This function filters and assigns microtubules whose trajectory or position qualifies
        them for inclusion in the middle region. The function utilizes input filaments, vertices,
        and poles to compute the selection. The MTs are specifically filtered based on their
        crossing status and proximity to the vertices and poles.

        Returns:
            list: A list of identifiers corresponding to microtubules assigned to the middle region.
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
            vertices_=np.vstack(self.vertices), mt_ends1=mts_plus, mt_ends2=mts_minus
        )
        if len(mid_mt_ids) == 0:
            return []

        mid_mt_fibers = self.filaments[np.isin(self.filaments[:, 0], mid_mt_ids)]
        mid_mt_ids = self.assign_mt_with_crossing(
            filaments=mid_mt_fibers, vertices_=np.vstack(self.vertices), class_=[0, 1]
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
        Assigns microtubules (MTs) to the intersection class through filtering and reclassification.
        The method isolates MTs that are not part of known groups such as kmts_id_1, kmts_id_2, and mid_mt_ids,
        determines those within bounding boxes, and further refines the classification based on crossing.

        Returns:
            list: A list of IDs for microtubules assigned to the intersection class.
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
            filaments=int_mt_fibers, vertices_=np.vstack(self.vertices), class_=[2, 3]
        )

        return list(int_mt_ids)

    def assign_to_bridge_mt(self) -> list:
        """
        Assigns filaments to bridge microtubules (MTs) based on specified classification
        parameters and conditions. Filaments that do not belong to kinetochore, middle,
        or interpolar MTs are filtered and assigned as bridge MTs after matching their
        properties with the respective vertices and classes provided.

        Return:
            list: A list of IDs corresponding to the filaments assigned as bridge MTs.
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
            vertices_=np.vstack(self.vertices),
            class_=[4, 5, 6],
        )
        return list(bridge_mt_ids)

    def assign_mt_with_crossing(self, filaments, vertices_, class_=[1]) -> np.ndarray:
        """
        Calculate microtubules crossing a surface based on specified criteria.

        This function identifies microtubules (MT) that cross a defined surface and
        categorizes them based on their crossing patterns. It evaluates each MT to
        determine if it satisfies the conditions of belonging to certain specified
        classes based on the number of crossing groups. If the `class_` parameter
        includes 0, MTs that do not cross the surface are also added to the result.

        Args:
            filaments (np.ndarray): A 2D NumPy array with the first column representing the
                microtubule IDs and the other columns representing their coordinates.
            vertices_ (np.ndarray): A 2D NumPy array representing the vertices of the surface
                mesh to evaluate crossing points against.
            class_ (list): A list of integers representing the classification of MTs
                based on the number of crossing groups they exhibit.

        Return:
            np.ndarray: A 1D NumPy array containing the IDs of the microtubules (MT) that
                meet the crossing criteria.
        """
        all_ids = np.unique(filaments[:, 0])

        """Calculate MT crossing the surface"""
        _, points_on_surface = points_on_mesh_knn(filaments[:, 1:], vertices_)

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

            if count_true_groups(f) in class_:
                ids.append(mt_id)

        if 0 in class_:
            for mt_id in all_ids:
                if mt_id not in mt_id_crossing:
                    ids.append(mt_id)
        return np.array(ids)

    def classified_MTs(self):
        """
        Classifies microtubules (MTs) into distinct categories based on their spatial characteristics and association
        with structural regions. The function performs multiple steps of analysis, including identifying the ends of
        filaments, categorizing Kinetochore MTs (KMTs), and further segregating filaments into Mid-MTs,
        Interdigitating-MTs, Bridging-MTs, and Single Microtubules (SMTs).

        Notes:
            The classification workflow includes multiple stages:

            1. Endpoint identification for filaments.
            2. Determination of associated KMTs based on filament-pole orientation, divided into separate pole-derived sets.
            3. Segregation of KMTs into inside and outside regions within each pole group.
            4. Assignment of non-KMTs into categories including Mid-MTs, Interdigitating-MTs, and Bridging-MTs.
            5. Selection of SMTs as the remaining microtubules after prior classifications.
            6. Correction of coordinate transformations to pixel size units for visualization.
        """
        """Get indices for ends"""
        self.plus_end_id, self.minus_end_id = self.get_filament_endpoints()

        """Get indices for KMTs"""
        self.kmts_id_1 = list(self.assign_to_kmts(filaments=self.filament_pole1, id_=0))
        self.kmts_id_2 = list(self.assign_to_kmts(filaments=self.filament_pole2, id_=1))

        self.kmts_inside_id_1, self.kmts_outside_id_1 = self.kmts_inside_outside(
            self.filaments[np.isin(self.filaments[:, 0], self.kmts_id_1)], id_=0
        )
        self.kmts_inside_id_1, self.kmts_outside_id_1 = list(
            self.kmts_inside_id_1
        ), list(self.kmts_outside_id_1)
        self.kmts_outside_id_2, self.kmts_inside_id_2 = self.kmts_inside_outside(
            self.filaments[np.isin(self.filaments[:, 0], self.kmts_id_2)], id_=1
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
        Retrieves a structured list of classified indices based on specific criteria
        used within the object.

        Return:
            list: A list containing grouped classified indices.
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
        Classifies and categorizes fibers based on their identification within specific regions
        or categories. The filtering is performed using the `np.isin` method to check
        membership of the provided filament IDs against specific ID lists.


        Return:
            list: A list of lists and arrays corresponding to different classified fibers.
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
        Retrieves the filaments.

        Return:
            np.ndarray: A numpy array containing the filaments.
        """
        return self.filaments

    def get_vertices(self, simplify=128) -> tuple[list, list]:
        """
        Retrieves the vertices and triangles of a 3D surface.

        Args:
            simplify (int): Value used to simplify the vertices and triangles data. If None, retrieves data from file.

        Return:
            tuple: A tuple containing a list of vertices and triangles.
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
        This method retrieves the poles coordinates.

        Return:
            np.ndarray: The poles coordinates
        """

        return self.poles
