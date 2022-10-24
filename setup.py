from setuptools import setup, find_packages
from tardis_dev.version import version

with open('./doc/source/README.rst') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as r:
    required = r.read().splitlines()

setup(
    author=["Robert Kiewisz", "Tristan Bepler"],
    author_email='rkiewisz@nysbc.com',
    python_requires='>=3.7',
    install_requires=required,
    classifiers=['Development Status :: Alpha Release',
                 'Intended Audience :: Developers/Research',
                 'Environment :: Console/WebApp',
                 'Environment :: GPU :: NVIDIA CUDA :: >=11.3',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Programming Language :: Python :: 3.7'],
    description="PyTorch segmentation of 2D/3D images such as electron tomography "
                "(ET), Cryo-EM or fluorescent microscopy data into 3D segmented "
                "point cloud.",
    entry_points={
        'console_scripts': [
            'tardis_cnn_train = tardis.train_spindletorch:main',
            'tardis_cnn_predict = tardis.predict_spindletorch:main',
            'tardis_postprocessing = tardis.spindletorch_postprocess:main',
            'tardis_pointcloud_train = tardis.train_DIST:main',
            'tardis_gf_score = tardis.DIST_score:main',
            'tardis_mt = tardis.predict_MTs:main',
        ],
    },
    license="MIT License",
    long_description_content_type='text/x-rst',
    long_description=readme,
    include_package_data=True,
    keywords=['spindletorch', 'semantic segmentation', 'point cloud segmentation',
              'MT segmentation', 'UNet', 'Unet3Plus'],
    name='tardis-dev',
    packages=find_packages(include=['tardis', 'tardis.*'],
                           exclude=['tests']),
    url='https://github.com/SMLC-NYSBC/tardis-pytorch',
    version=version,
)
