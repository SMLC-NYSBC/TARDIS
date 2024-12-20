#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
import time
from os import makedirs
from os.path import isdir, join
from shutil import rmtree
from typing import List

import numpy as np
import torch

from tardis_em.dist_pytorch.datasets.dataloader import build_dataset
from tardis_em.dist_pytorch.utils.segment_point_cloud import PropGreedyGraphCut
from tardis_em.cnn.datasets.build_dataset import (
    build_train_dataset,
)
from tardis_em.cnn.datasets.dataloader import PredictionDataset
from tardis_em.utils.errors import TardisError
from tardis_em.utils.load_data import load_image
from tardis_em.utils.logo import print_progress_bar, TardisLogo
from tardis_em.utils.metrics import AP, AUC, calculate_f1, IoU, mcov
from tardis_em.utils.predictor import Predictor


class CnnBenchmark:
    """
    Wrapper for CNN benchmark dataset

    Args:
        model (Predictor): Predictor class with evaluated model.
        dataset (str): Dataset name to evaluate model on.
        dir_s (str): Dataset directory.
        patch_size (int): Image patch size for CNN model.
        threshold (float): Threshold value used for prediction..
    """

    def __init__(
        self,
        model: Predictor,
        dataset: str,
        dir_s: str,
        patch_size: int,
        threshold: float,
    ):
        self.tardis_progress = TardisLogo()
        self.title = "TARDIS - CNN Benchmark"
        self.tardis_progress(title=self.title)

        self.model = model
        self.threshold = threshold
        self.metric = {
            "Acc": [],
            "Prec": [],
            "Recall": [],
            "F1": [],
            "AUC": [],
            "IoU": [],
            "AP": [],
        }

        self.data_set = dataset
        self.dir = dir_s
        if isdir(join(dir_s, "train")):
            rmtree(join(dir_s, "train"))

        makedirs(join(dir_s, "train", "imgs"))
        makedirs(join(dir_s, "train", "masks"))

        if self.data_set in ["MT", "Mem"]:
            build_train_dataset(
                dataset_dir=self.dir,
                circle_size=150,
                resize_pixel_size=25,
                trim_xy=patch_size,
                trim_z=patch_size,
            )
        else:
            TardisError(
                id_="",
                py="tardis_em/benchmarks/benchmarks.py",
                desc=f"Given data set {self.data_set} is not supporter!",
            )

        self.eval_data = PredictionDataset(img_dir=join(self.dir, "train", "imgs"))

    def _benchmark(self, logits: np.ndarray, target: np.ndarray):
        input = np.where(logits >= self.threshold, 1, 0).astype(np.uint8)

        # ACC, Prec, Recall, F1
        accuracy_score, precision_score, recall_score, F1_score = calculate_f1(
            input, target, False
        )
        self.metric["Acc"].append(accuracy_score)
        self.metric["Prec"].append(precision_score)
        self.metric["Recall"].append(recall_score)
        self.metric["F1"].append(F1_score)

        # AUC
        self.metric["AUC"].append(AUC(logits, target))

        # IoU
        self.metric["IoU"].append(IoU(input, target))

        # AP
        self.metric["AP"].append(AP(logits, target))

    def _predict(self, input: torch.Tensor) -> np.ndarray:
        return self.model.predict(input[None, :])

    def _output_metric(self):
        return {k: np.mean(v) for k, v in self.metric.items()}

    def __call__(self) -> dict:
        """
        Main call for benchmarking CNN.

        It takes model, predict given image and run benchmark on it to return
        standard metrics for CNN.
        """
        iter_ = 1
        self.tardis_progress(
            title=self.title,
            text_1=f"Running image segmentation benchmark on "
            f"{self.data_set} dataset",
            text_4="Benchmark: In progress...",
            text_5="Current Task: CNN Benchmark...",
            text_8=print_progress_bar(0, len(self.eval_data)),
        )

        for i in range(len(self.eval_data)):
            """Predict"""
            # Pick image['s]
            input, name = self.eval_data.__getitem__(i)

            if i == 0:
                start = time.time()
                input = self._predict(input)
                end = time.time()

                # Scale progress bar refresh to 10s
                iter_ = 10 // (end - start)
                if iter_ < 1:
                    iter_ = 1
            else:
                input = self._predict(input)

            target, _ = load_image(join(self.dir, "train", "masks", f"{name}_mask.tif"))
            target = target.astype(np.uint8)

            """Benchmark"""
            self._benchmark(input, target)

            metric_1 = (
                f'mAcc: {round(np.mean(self.metric["Acc"]), 2)}; '
                f'mPrec: {round(np.mean(self.metric["Prec"]), 2)}; '
                f'mRecall: {round(np.mean(self.metric["Recall"]), 2)}; '
                f'mF1: {round(np.mean(self.metric["F1"]), 2)}'
            )
            metric_2 = (
                f'mAUC: {round(np.mean(self.metric["AUC"]), 2)}; '
                f'mIoU: {round(np.mean(self.metric["IoU"]), 2)}; '
                f'mAP: {round(np.mean(self.metric["AP"]), 2)}'
            )

            if i % iter_ == 0:
                # Tardis progress bar update
                self.tardis_progress(
                    title=self.title,
                    text_1=f"Running image segmentation benchmark on "
                    f"{self.data_set} dataset",
                    text_4="Benchmark: In progress...",
                    text_5="Current Task: CNN Benchmark...",
                    text_6=metric_1,
                    text_7=metric_2,
                    text_8=print_progress_bar(i, len(self.eval_data)),
                )

        rmtree(join(self.dir, "train"))
        return self._output_metric()


class DISTBenchmark:
    """
    Wrapper for DIST benchmark dataset

    Args:
        model (Predictor): Predictor class with evaluated model.
        dataset (str): Dataset name to evaluate model on.
        dir_s (str): Dataset directory.
        points_in_patch (int): Max. number of points in sub-graphs.
        threshold (float): Threshold value used for prediction..
    """

    def __init__(
        self,
        model: Predictor,
        dataset: str,
        dir_s: str,
        points_in_patch: int,
        threshold: float,
    ):
        self.tardis_progress = TardisLogo()
        self.title = "TARDIS - DIST Benchmark"
        self.tardis_progress(title=self.title)

        self.model = model
        self.dir = dir_s
        self.data_set = dataset
        if self.data_set in ["MT", "Mem"]:
            self.sort = True
        else:
            self.sort = False

        self.threshold = threshold
        self.metric = {
            "IoU": [],  # Graph
            "AUC": [],  # Graph
            "mCov": [],  # Instance
            "mWCov": [],  # Instance
        }
        if dataset.endswith("rgb"):
            self.rgb = True
        else:
            self.rgb = False

        self.eval_data = build_dataset(
            dataset_type=dataset,
            dirs=[None, self.dir],
            max_points_per_patch=points_in_patch,
            benchmark=True,
        )

        if dataset in ["MT", "Mem"]:
            self.max_connections = 2
        else:
            self.max_connections = 4

    def _benchmark_graph(self, logits: np.ndarray, target: np.ndarray):
        input = np.where(logits >= self.threshold, 1, 0).astype(np.uint8)

        # IoU
        self.metric["IoU"].append(IoU(input, target, True))

        # AUC
        self.metric["AUC"].append(AUC(logits, target, True))

    def _benchmark_IS(
        self, logits: List[np.ndarray], coords: np.ndarray, output_idx: List[np.ndarray]
    ):
        # Graph cut
        GraphToSegment = PropGreedyGraphCut(
            threshold=self.threshold, connection=self.max_connections
        )
        input_IS = GraphToSegment.patch_to_segment(
            graph=logits, coord=coords[:, 1:], idx=output_idx, prune=5, sort=self.sort
        )

        # mCov and mWCov
        mCov, mwCov = mcov(input_IS, coords)
        self.metric["mCov"].append(mCov)
        self.metric["mWCov"].append(mwCov)

    def _predict(self, input, node=None):
        if node is not None:
            return self.model.predict(input, node)
        return self.model.predict(input)

    def _output_metric(self):
        return {k: np.mean(v) for k, v in self.metric.items()}

    def _update_metric_pg(self):
        mean_or_nan = lambda x: round(np.mean(x), 2) if len(x) != 0 else "nan"

        iou = mean_or_nan(self.metric["IoU"])
        auc = mean_or_nan(self.metric["AUC"])
        m_mcov = mean_or_nan(self.metric["mCov"])
        m_mwcov = mean_or_nan(self.metric["mWCov"])

        pg = f"IoU: {iou}; " f"AUC: {auc}; " f"mCov: {m_mcov}; " f"mWCov: {m_mwcov}"
        return pg

    def __call__(self):
        # Tardis progress bar update
        self.tardis_progress(
            title=self.title,
            text_1=f"Running point cloud segmentation benchmark on " f"{self.data_set}",
            text_4="Benchmark: In progress...",
            text_5="File: Nan",
            text_7="Current Task: DIST prediction...",
            text_8=print_progress_bar(0, len(self.eval_data)),
        )

        for i in range(len(self.eval_data)):
            """Predict"""
            (
                idx,
                coord_org,
                coords,
                nodes,
                target,
                output_idx,
                _,
            ) = self.eval_data.__getitem__(i)
            target = [t.cpu().detach().numpy() for t in target]
            output_idx = [o.cpu().detach().numpy() for o in output_idx]

            self.tardis_progress(
                title=self.title,
                text_1=f"Running point cloud segmentation benchmark on "
                f"{self.data_set}",
                text_4="Benchmark: In progress...",
                text_5=f"File: {idx}",
                text_7=self._update_metric_pg(),
                text_8="Current Task: DIST prediction...",
                text_9=print_progress_bar(i, len(self.eval_data)),
            )

            graphs = []
            for edge, graph, node in zip(coords, target, nodes):
                if self.rgb:
                    input = self._predict(edge[None, :], node[None, :])
                else:
                    input = self._predict(edge[None, :])
                graphs.append(input)

                """Benchmark Graph"""
                self._benchmark_graph(input, graph.astype(np.uint8))

            self.tardis_progress(
                title=self.title,
                text_1=f"Running point cloud segmentation benchmark on "
                f"{self.data_set}",
                text_4="Benchmark: In progress...",
                text_5=f"File: {idx}",
                text_7=self._update_metric_pg(),
                text_8="Current Task: DIST prediction...",
                text_9=print_progress_bar(i, len(self.eval_data)),
            )

            """Segment graphs"""
            self._benchmark_IS(graphs, coord_org, output_idx)

            self.tardis_progress(
                title=self.title,
                text_1=f"Running point cloud segmentation benchmark on "
                f"{self.data_set}",
                text_4="Benchmark: In progress...",
                text_5=f"File: {idx}",
                text_7=self._update_metric_pg(),
                text_8="Current Task: DIST prediction...",
                text_9=print_progress_bar(i, len(self.eval_data)),
            )

        return self._output_metric()
