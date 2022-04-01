from setuptools import setup, find_packages
from tardis.version import version

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('docs/HISTORY.rst') as history_file:
    history = history_file.read()

setup(
    author=["Robert Kiewisz", "Tristan Bepler"],
    author_email='rkiewisz@nysbc.com',
    python_requires='>=3.8',
    requirements=[],
    classifiers=[
        'Development Status :: Drafting Alpha Release',
        'Intended Audience :: Developers/Research',
        'Environment :: Console/WebApp',
        'Environment :: GPU :: NVIDIA CUDA :: 11.1',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ],
    description="PyTorch segmentation of 2D/3D images such as electron tomography "
                "(ET), Cryo-EM or fluorescent microscopy data into 3D segmented "
                "point cloud.",
    entry_points={
        'console_scripts': [
            'tardis_cnn_train=tardis.train_image_segmentation:main', 
            ],
    },
    license="MIT License",
    long_description_content_type='text/x-rst',
    long_description=readme,
    include_package_data=True,
    keywords=['spindletorch', 'semantic segmentation', 'point cloud segmentation',
              'MT segmentation', 'UNet', 'Unet3Plus'],
    name='tardis',
    longname='Transformer And Rapid Dimensionless Instance Segmentation',
    packages=find_packages(include=['tardis']),
    url='https://github.com/SMLC-NYSBC/tardis',
    version=version,
)
