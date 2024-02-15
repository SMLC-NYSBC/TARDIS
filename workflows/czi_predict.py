from os.path import join

import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config
from cryoet_data_portal import Client, Tomogram
import numpy as np
from s3transfer import S3Transfer

from tardis_em.utils.logo import TardisLogo


def get_from_aws(url):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3.download_file('cryoet-data-portal-public',
                     url, url.split('/')[-1])


def upload_to_aws(aws_key, aws_secret, bucket, id_, data_name, file_name):
    client = boto3.client('s3',
                          aws_access_key_id=aws_key,
                          aws_secret_access_key=aws_secret)
    transfer = S3Transfer(client)
    transfer.upload_file(file_name,
                         bucket,
                         join('robert_kiewisz_tardis_01_2024',
                              id_,
                              data_name,
                              file_name.split('/')[-1]))


def main(aws_dir: str, bucket: str, allocate_gpu=(5, 6, 7)):
    terminal = TardisLogo()

    aws = np.genfromtxt(aws_dir, delimiter=',')
    aws_key = aws[0]
    aws_secret = aws[1]

    client = Client()

    # Get all tomograms with voxel spacing <= 10
    tomos = list(Tomogram.find(client, ))

    # S3 URIs for MRCs
    s3mrc = [t.s3_mrc_scale0 for t in tomos]
    id_ = [s[31:36] for s in s3mrc]
    data = [s.split('/')[4] for s in s3mrc]
    px = [float(s.split('/')[6][12:]) for s in s3mrc]


if __name__ == "__main__":
    main()
