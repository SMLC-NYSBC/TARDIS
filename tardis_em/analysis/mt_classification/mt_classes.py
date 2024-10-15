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

    def get_vertices_file(self, dir_, simplify):
        _, _, vertices, self.triangles = load_am_surf(dir_, simplify_=simplify)

        return vertices

    @staticmethod
    def get_filament_file(dir_):
        return ImportDataFromAmira(src_am=dir_).get_segmented_points()

    @staticmethod
    def get_poles_file(dir_):
        return ImportDataFromAmira(src_am=dir_).get_vertex()

    def correct_data(self):
        """Correct the coordinates based on pixel size."""
        self.min_coords = self.filaments[:, 1:].min(axis=0)

        # Efficient normalization using broadcasting
        self.filaments[:, 1:] = (self.filaments[:, 1:] -self. min_coords) / self.pixel_size
        self.filaments = resample_filament(self.filaments, 1)

        # Normalize vertices and poles
        for i in range(len(self.vertices)):
            self.vertices[i] = (self.vertices[i] - self.min_coords) / self.pixel_size

        self.poles = ((self.poles - self.min_coords) / self.pixel_size).astype(np.int32)
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

    def get_filament_endpoints(self):
        """Returns the start and end points of each filament."""
        ids = self.filaments[:, 0]
        unique_ids, index_starts, counts = np.unique(
            ids, return_index=True, return_counts=True
        )
        end_indices = index_starts + counts - 1

        return index_starts, end_indices

    def assign_to_kmts(self, filaments, id_=0):
        """Assign filaments to KMTs based on distance and surface crossing."""
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

    def kmts_inside_outside(self, kmt_proposal, id_=0):
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

    def assign_to_mid_mt(self):
        """Select ends and fibers within BB"""
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

    def assign_to_int_mt(self):
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

    def assign_to_bridge_mt(self):
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

    def assign_mt_with_crossing(self, filaments, vertices_, class_=[1]):
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

    def get_classified_indices(self):
        return [
            [self.kmts_inside_id_1, self.kmts_outside_id_1],
            [self.kmts_inside_id_2, self.kmts_outside_id_2],
            self.mid_mt_ids,
            self.int_mt_ids,
            self.brg_mt_ids,
            self.smt_ids,
        ]

    def get_classified_fibers(self):
        self.filaments[:, 1:] = (self.filaments[:, 1:] + self.min_coords) * self.pixel_size

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

    def get_filaments(self):
        return self.filaments

    def get_vertices(self, simplify=128):
        if simplify is not None:
            return self.vertices, self.triangles
        else:
            _, _, vertices, self.triangles = self.get_vertices_file(
                self.surfaces, simplify
            )
            return vertices, self.triangles

    def get_poles(self):
        return self.poles
