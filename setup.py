#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################
from setuptools import find_packages, setup

from tardis.version import version

with open("docs/source/README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as r:
    required = r.read().splitlines()

setup(
    author="Robert Kiewisz, Tristan Bepler",
    author_email="rkiewisz@nysbc.com",
    python_requires=">=3.7, <3.11",  # 3.11 support soon
    install_requires=required,
    dependency_links=["https://download.pytorch.org/whl/cu117"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers/Research",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA :: >=11.3",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="PyTorch segmentation of 2D/3D images such as electron tomography "
                "(ET), Cryo-EM or fluorescent microscopy data into 3D segmented "
                "point cloud.",
    entry_points={
        "console_scripts": [
            "tardis_cnn_train = tardis.train_spindletorch:main",
            "tardis_dist_train = tardis.train_DIST:main",
            "tardis_mt = tardis.predict_mt:main",
            "tardis_mem = tardis.predict_mem:main",
            "tardis_compare_sg = tardis.compare_spatial_graphs:main",
            "tardis_benchmark = tardis.benchmarks.benchmarks:main",
        ],
    },
    license="MIT License",
    long_description_content_type="text/x-rst",
    long_description=readme,
    include_package_data=True,
    keywords=[
        "spindletorch",
        "semantic segmentation",
        "point cloud segmentation",
        "MT segmentation",
        "UNet",
        "Unet3Plus",
        "FNet",
    ],
    name="tardis_pytorch",
    packages=find_packages(),
    url="https://github.com/SMLC-NYSBC/tardis-pytorch",
    version=version,
)
