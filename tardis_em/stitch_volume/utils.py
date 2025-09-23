#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2025                                            #
#######################################################################

from os import listdir
from os.path import isfile, join, split, splitext


def sort_tomogram_files(path):
    image_exts = {".tif", ".tiff", ".mrc", ".rec", ".am"}
    coord_exts = {".csv", ".am"}

    images = []
    coordinates = []

    for fname in listdir(path):
        fpath = join(path, fname)
        if not isfile(fpath):
            continue

        ext = splitext(fname)[1].lower()

        # Normal image types (non-.am)
        if ext in image_exts and ext != ".am":
            images.append(fpath)
            continue

        # Normal coordinate types (non-.am)
        if ext in coord_exts and ext != ".am":
            coordinates.append(fpath)
            continue

        # Handle .am files → need to peek inside
        if ext == ".am":
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    first_lines = [next(f).strip() for _ in range(4)]

                    if len(first_lines) != 0:
                        first_lines = [line for line in first_lines if line.strip()]
            except Exception:
                continue  # skip unreadable files

            if first_lines[0].startswith("# AmiraMesh BINARY-LITTLE-ENDIAN"):
                # This is an image-like .am file
                images.append(fpath)
            else:
                # Otherwise treat as coordinates
                coordinates.append(fpath)

    # There can be more files than needed, sort by images and find corresponding coord files
    img_path_list, coord_path_list = [], []
    for img in sorted(images):
        img_path_list.append(img)

        img = splitext(split(img)[-1])[0].lower()

        matches = [p for p in coordinates if img in splitext(split(p)[-1])[0].lower()]

        if len(matches) == 0:
            coord_path_list.append(None)
        else:
            coord_path_list.append(matches[0])

    return img_path_list, coord_path_list
