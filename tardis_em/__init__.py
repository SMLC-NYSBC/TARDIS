from tardis_em._version import version

__version__ = version

format_choices = [
    f"{prefix}_{suffix}"
    for prefix in ["None", "am", "mrc", "tif", "npy"]
    for suffix in ["None", "am", "mrc", "tif", "npy", "amSG", "csv", "stl"]
]
