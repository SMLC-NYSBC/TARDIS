#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import asyncio
import shutil
from os.path import join

import boto3
import click
import pandas as pd
import torch
import tifffile.tifffile as tif

from tardis_em.cnn.data_processing.stitch import StitchImages
from tardis_em.cnn.data_processing.trim import trim_with_stride
from tardis_em.cnn.datasets.dataloader import PredictionDataset
from tardis_em.cnn.utils.utils import scale_image
from tardis_em.dist_pytorch.datasets.patches import PatchDataSet
from tardis_em.dist_pytorch.utils.build_point_cloud import BuildPointCloud
from tardis_em.dist_pytorch.utils.segment_point_cloud import PropGreedyGraphCut
from tardis_em import version
from tardis_em.utils.load_data import load_image, mrc_read_header
from botocore import UNSIGNED
from botocore.config import Config
from cryoet_data_portal import Client, Tomogram
import numpy as np
from s3transfer import S3Transfer

from tardis_em.utils.logo import TardisLogo, print_progress_bar
from tardis_em.utils.device import get_device
from tardis_em.utils.predictor import Predictor
from tardis_em.utils.normalization import RescaleNormalize, MeanStdNormalize
from tardis_em.utils.export_data import to_mrc
from shutil import rmtree

def get_from_aws(url):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3.download_file("cryoet-data-portal-public", url, url.split("/")[-1])


def upload_to_aws(aws_key, aws_secret, bucket, id_, data_name, file_name):
    client = boto3.client(
        "s3", aws_access_key_id=aws_key, aws_secret_access_key=aws_secret
    )
    transfer = S3Transfer(client)
    transfer.upload_file(
        file_name,
        bucket,
        join("robert_kiewisz_tardis_01_2024", id_, data_name, file_name),
    )
    rmtree(file_name)


class ProcessTardisForCZI:
    def __init__(self, allocate_gpu: int):
        self.device = get_device(allocate_gpu)
        self.normalize_px = 15
        self.image_stitcher = StitchImages()
        self.post_processes = BuildPointCloud()

        # Normalize histogram
        self.normalize = RescaleNormalize(clip_range=(1, 99))
        self.mean_std = MeanStdNormalize()

        # Load CNN model
        self.model_cnn = Predictor(
            checkpoint=None,
            network="unet",
            subtype="32",
            model_type="membrane_3d",
            img_size=128,
            sigmoid=False,
            device=self.device,
        )

        self.model_dist = Predictor(
            checkpoint=None,
            network="dist",
            subtype="triang",
            model_type="3d",
            device=self.device,
        )

    def predict_cnn(self, dataloader):
        for j in range(len(dataloader)):
            input_, name = dataloader.__getitem__(j)
            input_ = self.model_cnn.predict(input_[None, :], rotate=True)
            tif.imwrite(join(self.output, f"{name}.tif"), input_)

    def __call__(self, data: np.ndarray, px: float, header, name):
        data = self.normalize(self.mean_std(data)).astype(np.float32)
        tif.imwrite(name + ".tif", data.sum(0).astype(np.uint16))

        # Sanity check image histogram
        if not data.min() >= -1 or not data.max() <= 1:  # Image not between in -1 and 1
            if data.min() >= 0 and data.max() <= 1:
                data = (data - 0.5) * 2  # shift to -1 - 1
            elif data.min() >= 0 and data.max() <= 255:
                data = data / 255  # move to 0 - 1
                data = (data - 0.5) * 2  # shift to -1 - 1

        scale_factor = px / self.normalize_px
        org_shape = data.shape
        scale_shape = np.multiply(org_shape, scale_factor).astype(np.int16)
        scale_shape = [int(i) for i in scale_shape]

        trim_with_stride(
            image=data,
            scale=scale_shape,
            trim_size_xy=128,
            trim_size_z=128,
            output=join("temp", "Patches"),
            image_counter=0,
            clean_empty=False,
            stride=round(128 * 0.125),
        )

        self.predict_cnn(
            dataloader=PredictionDataset(join("temp", "Patches", "imgs")),
        )
        data = self.image_stitcher(
            image_dir=join("temp", "Predictions"), mask=False, dtype=np.float32
        )[: scale_shape[0], : scale_shape[1], : scale_shape[2]]
        data, _ = scale_image(image=data, scale=org_shape, nn=True)
        data = torch.sigmoid(torch.from_numpy(data)).cpu().detach().numpy()

        data = np.where(data > 0.25, 1, 0).astype(np.uint8)
        to_mrc(data, px, name + "_semantic.mrc", header)
        tif.imwrite(name + "_semantic.tif", data.sum(0).astype(np.uint8))

        _, pc_ld = BuildPointCloud().build_point_cloud(
            image=data, EDT=False, down_sampling=5, as_2d=False
        )

        coords_df, _, output_idx, _ = PatchDataSet(
            max_number_of_points=900, graph=False
        ).patched_dataset(coord=pc_ld)
        x = []
        with torch.no_grad():
            for i in coords_df:
                x.append(
                    self.model_cnn.predict(x=i[None, :].to(self.device), y=None)
                    .cpu()
                    .detach()
                    .numpy()[0, 0, :]
                )

        pc = PropGreedyGraphCut(threshold=0.5, connection=8).patch_to_segment(
            graph=x,
            coord=pc_ld,
            idx=output_idx,
            sort=False,
            prune=15,
        )
        pc = pd.DataFrame(pc)
        pc.to_csv(
            name + "_instance.csv",
            header=["IDs", "X [A]", "Y [A]", "Z [A]"],
            index=False,
            sep=",",
        )


@click.option(
    "-aws",
    "--aws_dir",
    type=str,
    show_default=True,
)
@click.option(
    "-bucket",
    "--bucket",
    default='cryoet-data-portal-public',
    type=str,
    show_default=True,
)
@click.option(
    "-gpu",
    "--allocate_gpu",
    default=7,
    type=str,
    show_default=True,
)
@click.version_option(version=version)
async def main(aws_dir: str, bucket: str, allocate_gpu: int):
    terminal = TardisLogo()
    terminal(title='Predict CIZ-Cryo-EM data-port datasets')
    process_tardis = ProcessTardisForCZI(allocate_gpu)

    aws = np.genfromtxt(aws_dir, delimiter=",")
    aws_key = aws[0]
    aws_secret = aws[1]

    client = Client()

    # Get all tomograms with voxel spacing <= 10
    tomos = list(
        Tomogram.find(
            client,
        )
    )

    # S3 URIs for MRCs
    urls = [t.s3_mrc_scale0 for t in tomos]
    folder_names = [s[31:36] for s in urls]
    file_names = [s.split("/")[4] for s in urls]
    czi_px = [float(s.split("/")[6][12:]) for s in urls]

    terminal(title='Predict CIZ-Cryo-EM data-port datasets',
             text_1='Prediction:',
             text_3=f'Dataset: []',
             text_5=print_progress_bar(0, len(urls)),)

    tasks = []
    next_download = asyncio.create_task(get_from_aws(urls[0][31:]))
    for idx, (url, folder_name, file_name, px_czi) in enumerate(
            zip(urls, folder_names, file_names, czi_px)
    ):
        terminal(title='Predict CIZ-Cryo-EM data-port datasets',
                 text_1='Prediction:',
                 text_3=f'Dataset: [{file_name}]',
                 text_5=print_progress_bar(idx, len(urls)), )

        # Wait for the current download to complete before processing
        await next_download

        data, px = load_image(file_name, False)
        header = mrc_read_header(file_name)

        if px != px_czi:
            px = px_czi

        process_task = asyncio.create_task(
            asyncio.get_running_loop().run_in_executor(
                None, process_tardis, data, px, header, file_name[:-4]
            )
        )

        # If there's a next file, start downloading it now
        if url != urls[-1]:
            next_download = asyncio.create_task(get_from_aws(url[31:]))

        await process_task

        # Upload the processed files
        tasks += [
            asyncio.create_task(
                upload_to_aws(
                    aws_key,
                    aws_secret,
                    bucket,
                    folder_names,
                    file_name[:-4],
                    file_name[:-4] + "_semantic.mrc",
                )
            ),
            asyncio.create_task(
                upload_to_aws(
                    aws_key,
                    aws_secret,
                    bucket,
                    folder_names,
                    file_name[:-4],
                    file_name[:-4] + "_instance.csv",
                )
            ),
        ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    main()
