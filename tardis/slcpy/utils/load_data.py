import struct
from collections import namedtuple
from os.path import isfile
from typing import Optional

import numpy as np
import open3d as o3d
import tifffile.tifffile as tif
from sklearn.neighbors import KDTree


class ImportDataFromAmira:
    """
    HANDLER for loading 3D .AM DATA

    Loading of Amira 3D spatial graph and/or image data (.tif/.mrc).
    In case of including image data, coordinates from spatial graph are
    corrected for Amira transformation.
    Ensuring matching of the coordinates with image data

    Args:
        src_am: Source of the spatial graph in ASCII format
        src_img: Source of the 3D .tif file
    """

    def __init__(self,
                 src_am: str,
                 src_img: Optional[str] = None):
        self.src_img = src_img
        self.src_am = src_am

        if self.src_img is not None:
            if not self.src_img[-3:] == '.am':
                raise Warning("Not a .am file...")

            if src_img.split('/')[-1:][:-3] != src_am.split('/')[-1:][:-20]:
                raise Warning(f'Image file {src_img} has wrong extension for {src_am}!')

            try:
                # Image file [Z x Y x X]
                self.image, self.pixel_size, _, self.transformation = import_am(src_img)
            except RuntimeWarning:
                raise Warning("Directory or input .am image file is not correct...")
        else:
            self.pixel_size = 1
        self.spatial_graph = open(src_am,
                                  "r",
                                  encoding="iso-8859-1").read().split("\n")
        self.spatial_graph = [x for x in self.spatial_graph if x != '']

    def get_segments(self):
        # Find line starting with EDGE { int NumEdgePoints }
        segments = str([word for word in self.spatial_graph if
                        word.startswith('EDGE { int NumEdgePoints }')])

        segment_start = "".join((ch if ch in "0123456789" else " ")
                                for ch in segments)
        segment_start = [int(i) for i in segment_start.split()]

        # Find in the line directory that starts with @..
        try:
            segment_start = int(self.spatial_graph.index("@" + str(segment_start[0]))) + 1
        except ValueError:
            segment_start = int(self.spatial_graph.index("@" + str(segment_start[0]) + " ")) + 1
        # Find line define EDGE ... <- number indicate number of segments
        segments = str(
            [word for word in self.spatial_graph if word.startswith('define EDGE')])

        segment_finish = "".join(
            (ch if ch in "0123456789" else " ") for ch in segments)
        segment_finish = [int(i) for i in segment_finish.split()]
        segment_no = int(segment_finish[0])
        segment_finish = segment_start + int(segment_finish[0])

        # Select all lines between @.. (+1) and number of segments
        segments = self.spatial_graph[segment_start:segment_finish]
        segments = [i.split(' ')[0] for i in segments]

        # return an array of number of points belonged to each segment
        df = np.zeros((segment_no, 1), dtype="int")
        df[0:segment_no, 0] = [int(i) for i in segments]

        return df

    def __find_points(self):
        # Find line starting with POINT { float[3] EdgePointCoordinates }
        points = str([word for word in self.spatial_graph
                      if word.startswith('POINT { float[3] EdgePointCoordinates }')])

        # Find in the line directory that starts with @..
        points_start = "".join((ch if ch in "0123456789" else " ")
                               for ch in points)
        points_start = [int(i) for i in points_start.split()]
        # Find line that start with the directory @.. and select last one
        try:
            points_start = int(self.spatial_graph.index("@" + str(points_start[1]))) + 1
        except ValueError:
            points_start = int(self.spatial_graph.index("@" + str(points_start[1]) + " ")) + 1

        # Find line define POINT ... <- number indicate number of points
        points = str([word for word in self.spatial_graph
                      if word.startswith('define POINT')])

        points_finish = "".join(
            (ch if ch in "0123456789" else " ") for ch in points)
        points_finish = [int(i) for i in points_finish.split()][0]
        points_no = points_finish
        points_finish = points_start + points_finish

        # Select all lines between @.. (-1) and number of points
        points = self.spatial_graph[points_start:points_finish]

        # return an array of all points coordinates in pixel
        df = np.zeros((points_no, 3), dtype="float")
        for j in range(3):
            coord = [i.split(' ')[j] for i in points]
            df[0:points_no, j] = [float(i) for i in coord]

        return df

    def __read_am_transformation(self):
        """
        This method read the header of ET (.am) file and determines global
        transformation for all coordinates
        !DEPRECIATED!
        """
        with open(self.src_img, "r", encoding="iso-8859-1") as et:
            lines_in_et = et.read(50000).split("\n")

        transformation_list = str([word for word in lines_in_et
                                   if word.startswith('    BoundingBox')]).split(" ")

        trans_x, trans_y, trans_z = (float(transformation_list[5]),
                                     float(transformation_list[7]),
                                     float(transformation_list[9]))
        return trans_x, trans_y, trans_z

    def get_points(self):
        """
        Output for point cloud as array of a shape [X, Y, Z]
        """
        if self.src_img is None:
            self.transformation = [0, 0, 0]
        points_coord = self.__find_points()

        points_coord[:, 0] = points_coord[:, 0] - self.transformation[0]
        points_coord[:, 1] = points_coord[:, 1] - self.transformation[1]
        points_coord[:, 2] = points_coord[:, 2] - self.transformation[2]

        return points_coord / self.pixel_size

    def get_segmented_points(self):
        """
        Output for segmented point cloud as array of a shape [ID, X, Y, Z]
        """
        points = self.get_points()
        segments = self.get_segments()

        segmentation = np.zeros((points.shape[0], ))
        id = 0
        idx = 0
        for i in segments:
            segmentation[id:(id + int(i))] = idx

            idx += 1
            id += int(i)

        return np.stack((segmentation,
                         points[:, 0],
                         points[:, 1],
                         points[:, 2])).T

    def get_image(self):
        return self.image, self.pixel_size

    def get_pixel_size(self):
        return self.pixel_size


def import_tiff(img: str,
                dtype=np.uint8):
    """
    Default import for tif files

    Args:
        img: x
        dtype: Type of output data
    Return:
        image: Image array of [Z, Y, X] shape
        pixel_size: 1
    """
    if not isfile(img):
        raise Warning("Indicated .tif file does not exist...")

    return np.array(tif.imread(img), dtype=dtype), 1


def import_mrc(img: str):
    """
    DEFAULT IMPORT FOR .mrc/.rec files

    Read out for MRC2014 files with

    Args:
        img: Source of image file

    Returns:
        image: Image array of [Z, Y, X] shape
        pixel_size: float value of the pixel size
    """
    if not isfile(img):
        raise Warning("Indicated .mrc file does not exist...")

    header = mrc_header(img)

    pixel_size = round(header.xlen / header.nx, 3)
    dtype = get_mode(header.mode, header.amin)
    nz, ny, nx = header.nz, header.ny, header.nx

    if nz == 1:
        image = np.fromfile(img, dtype=dtype)[1024:].reshape((ny, nx))
    else:
        try:
            image = np.fromfile(img, dtype=dtype)[1024:].reshape((nz, ny, nx))
        except:
            try:
                image = np.fromfile(img, dtype=dtype)[512:].reshape((nz, ny, nx))
            except:
                image = np.fromfile(img, dtype=dtype)[256:].reshape((nz, ny, nx))
    return image, pixel_size


def mrc_header(img: str):
    # int nx
    # int ny
    # int nz
    fstr = '3i'
    names = 'nx ny nz'

    # int mode
    fstr += 'i'
    names += ' mode'

    # int nxstart
    # int nystart
    # int nzstart
    fstr += '3i'
    names += ' nxstart nystart nzstart'

    # int mx
    # int my
    # int mz
    fstr += '3i'
    names += ' mx my mz'

    # float xlen
    # float ylen
    # float zlen
    fstr += '3f'
    names += ' xlen ylen zlen'

    # float alpha
    # float beta
    # float gamma
    fstr += '3f'
    names += ' alpha beta gamma'

    # int mapc
    # int mapr
    # int maps
    fstr += '3i'
    names += ' mapc mapr maps'

    # float amin
    # float amax
    # float amean
    fstr += '3f'
    names += ' amin amax amean'

    # int ispg
    # int next
    # short creatid
    fstr += '2ih'
    names += ' ispg next creatid'

    # pad 30 (extra data)
    # [98:128]
    fstr += '30x'

    # short nint
    # short nreal
    fstr += '2h'
    names += ' nint nreal'

    # pad 20 (extra data)
    # [132:152]
    fstr += '20x'

    # int imodStamp
    # int imodFlags
    fstr += '2i'
    names += ' imodStamp imodFlags'

    # short idtype
    # short lens
    # short nd1
    # short nd2
    # short vd1
    # short vd2
    fstr += '6h'
    names += ' idtype lens nd1 nd2 vd1 vd2'

    # float[6] tiltangles
    fstr += '6f'
    names += ' tilt_ox tilt_oy tilt_oz tilt_cx tilt_cy tilt_cz'

    # NEW-STYLE MRC image2000 HEADER - IMOD 2.6.20 and above
    # float xorg
    # float yorg
    # float zorg
    # char[4] cmap
    # char[4] stamp
    # float rms
    fstr += '3f4s4sf'
    names += ' xorg yorg zorg cmap stamp rms'

    # int nlabl
    # char[10][80] labels
    fstr += 'i800s'
    names += ' nlabl labels'

    header_struct = struct.Struct(fstr)
    MRCHeader = namedtuple('MRCHeader', names)

    with open(img, 'rb') as f:
        header = f.read(1024)

    return MRCHeader._make(header_struct.unpack(header))


def get_mode(mode, amin):
    """
    File formats for MRC2014

    Args:
        mode: MRC2014 mode
        amin: Optional min value to detect uint8
    """
    if mode == 0:
        if amin >= 0:
            dtype = np.uint8  # Unassigned 8-bit integer (-128 to 127)
        elif amin < 0:
            dtype = np.int8  # Signed 8-bit integer (0 - 254)
    elif mode == 1:
        dtype = np.int16  # Signed 16-bit integer
    elif mode == 2:
        dtype = np.float32  # Signed 32-bit real
    elif mode == 3:
        dtype = '2h'  # Complex 16-bit integers
    elif mode == 4:
        dtype = np.complex64  # Complex 32-bit reals
    elif mode == 6:
        dtype = np.uint16  # Unassigned int16
    elif mode == 12:
        dtype == np.float16  # Signed 16-bit half-precision real
    elif mode == 16:
        dtype = '3B'  # RGB values
    elif mode == 101:
        raise Exception('4 bit .mrc file are not supported. Ask Dev if you need it!')
    else:
        raise Exception('Unknown dtype mode:' + str(mode))

    return dtype


def import_am(img: str):
    """
    Default import for .am binary files

    Args:
        img: Source of image file
    """
    if not isfile(img):
        raise Warning(f"Indicated .am {img} file does not exist...")

    am = open(img, 'r', encoding="iso-8859-1").read(8000)
    assert '# AmiraMesh BINARY' in am, \
        f'{img} file is not Amira binary image file!'

    size = [word for word in am.split('\n') if word.startswith(
            'define Lattice ')][0][15:].split(" ")

    nx, ny, nz = int(size[0]), int(size[1]), int(size[2])

    # Fix for ET that were trimmed
    # Trimmed ET boundarybox has wrong size
    bb = str([word for word in am.split('\n') if word.startswith('    BoundingBox')]).split(" ")

    if len(bb) == 0:
        physical_size = np.array((float(bb[6]),
                                  float(bb[8]),
                                  float(bb[10][:-3])))
        binary_start = str.find(am, "\n@1\n") + 4
    else:
        am = open(img, 'r', encoding="iso-8859-1").read(20000)
        bb = str([word for word in am.split('\n') if word.startswith('    BoundingBox')]).split(" ")

        physical_size = np.array((float(bb[6]),
                                  float(bb[8]),
                                  float(bb[10][:-3])))

        transformation = np.array((float(bb[5]),
                                   float(bb[7]),
                                   float(bb[9])))
        binary_start = str.find(am, "\n@1\n") + 4

    try:
        coordinate = str([word for word in am.split('\n') if word.startswith('        Coordinates')]).split(" ")[9][1:2]
    except IndexError:
        coordinate = None

    if coordinate == 'm':  # Bring meter to angstrom
        pixel_size = ((physical_size[0] - transformation[0]) / (nx - 1)) * 10000000000
    else:
        pixel_size = (physical_size[0] - transformation[0]) / (nx - 1)
    pixel_size = round(pixel_size, 3)

    img = np.fromfile(img, dtype=np.uint8)

    if nz == 1:
        return img[binary_start:-1].reshape((ny, nx)), pixel_size, physical_size, transformation
    else:
        return img[binary_start:-1].reshape((nz, ny, nx)), pixel_size, physical_size, transformation


def load_ply(ply,
             downsample: Optional[None] = 0.035,
             scannet_data=True):
    """
    Loader for .ply files. .ply converted to point cloud and colors are used as labeling

    Args:
        ply: Directory for .ply file
        downsampling: Float value for .ply downsampling
    Return:
        np.ndarray of shape [Length x Dimension] dim are [L x X x Y x Z]
    """
    pcd = o3d.io.read_point_cloud(ply)
    label_uniq = np.unique(np.asarray(pcd.colors), axis=0)

    coord_org = np.asarray(pcd.points)
    label_org = np.asarray(pcd.colors)

    SCANNET_COLOR_MAP_200 = {
        0: (0., 0., 0.),
        1: (174., 199., 232.),
        2: (188., 189., 34.),
        3: (152., 223., 138.),
        4: (255., 152., 150.),
        5: (214., 39., 40.),
        6: (91., 135., 229.),
        7: (31., 119., 180.),
        8: (229., 91., 104.),
        9: (247., 182., 210.),
        10: (91., 229., 110.),
        11: (255., 187., 120.),
        13: (141., 91., 229.),
        14: (112., 128., 144.),
        15: (196., 156., 148.),
        16: (197., 176., 213.),
        17: (44., 160., 44.),
        18: (148., 103., 189.),
        19: (229., 91., 223.),
        21: (219., 219., 141.),
        22: (192., 229., 91.),
        23: (88., 218., 137.),
        24: (58., 98., 137.),
        26: (177., 82., 239.),
        27: (255., 127., 14.),
        28: (237., 204., 37.),
        29: (41., 206., 32.),
        31: (62., 143., 148.),
        32: (34., 14., 130.),
        33: (143., 45., 115.),
        34: (137., 63., 14.),
        35: (23., 190., 207.),
        36: (16., 212., 139.),
        38: (90., 119., 201.),
        39: (125., 30., 141.),
        40: (150., 53., 56.),
        41: (186., 197., 62.),
        42: (227., 119., 194.),
        44: (38., 100., 128.),
        45: (120., 31., 243.),
        46: (154., 59., 103.),
        47: (169., 137., 78.),
        48: (143., 245., 111.),
        49: (37., 230., 205.),
        50: (14., 16., 155.),
        51: (196., 51., 182.),
        52: (237., 80., 38.),
        54: (138., 175., 62.),
        55: (158., 218., 229.),
        56: (38., 96., 167.),
        57: (190., 77., 246.),
        58: (208., 49., 84.),
        59: (208., 193., 72.),
        62: (55., 220., 57.),
        63: (10., 125., 140.),
        64: (76., 38., 202.),
        65: (191., 28., 135.),
        66: (211., 120., 42.),
        67: (118., 174., 76.),
        68: (17., 242., 171.),
        69: (20., 65., 247.),
        70: (208., 61., 222.),
        71: (162., 62., 60.),
        72: (210., 235., 62.),
        73: (45., 152., 72.),
        74: (35., 107., 149.),
        75: (160., 89., 237.),
        76: (227., 56., 125.),
        77: (169., 143., 81.),
        78: (42., 143., 20.),
        79: (25., 160., 151.),
        80: (82., 75., 227.),
        82: (253., 59., 222.),
        84: (240., 130., 89.),
        86: (123., 172., 47.),
        87: (71., 194., 133.),
        88: (24., 94., 205.),
        89: (134., 16., 179.),
        90: (159., 32., 52.),
        93: (213., 208., 88.),
        95: (64., 158., 70.),
        96: (18., 163., 194.),
        97: (65., 29., 153.),
        98: (177., 10., 109.),
        99: (152., 83., 7.),
        100: (83., 175., 30.),
        101: (18., 199., 153.),
        102: (61., 81., 208.),
        103: (213., 85., 216.),
        104: (170., 53., 42.),
        105: (161., 192., 38.),
        106: (23., 241., 91.),
        107: (12., 103., 170.),
        110: (151., 41., 245.),
        112: (133., 51., 80.),
        115: (184., 162., 91.),
        116: (50., 138., 38.),
        118: (31., 237., 236.),
        120: (39., 19., 208.),
        121: (223., 27., 180.),
        122: (254., 141., 85.),
        125: (97., 144., 39.),
        128: (106., 231., 176.),
        130: (12., 61., 162.),
        131: (124., 66., 140.),
        132: (137., 66., 73.),
        134: (250., 253., 26.),
        136: (55., 191., 73.),
        138: (60., 126., 146.),
        139: (153., 108., 234.),
        140: (184., 58., 125.),
        141: (135., 84., 14.),
        145: (139., 248., 91.),
        148: (53., 200., 172.),
        154: (63., 69., 134.),
        155: (190., 75., 186.),
        156: (127., 63., 52.),
        157: (141., 182., 25.),
        159: (56., 144., 89.),
        161: (64., 160., 250.),
        163: (182., 86., 245.),
        165: (139., 18., 53.),
        166: (134., 120., 54.),
        168: (49., 165., 42.),
        169: (51., 128., 133.),
        170: (44., 21., 163.),
        177: (232., 93., 193.),
        180: (176., 102., 54.),
        185: (116., 217., 17.),
        188: (54., 209., 150.),
        191: (60., 99., 204.),
        193: (129., 43., 144.),
        195: (252., 100., 106.),
        202: (187., 196., 73.),
        208: (13., 158., 40.),
        213: (52., 122., 152.),
        214: (128., 76., 202.),
        221: (187., 50., 115.),
        229: (180., 141., 71.),
        230: (77., 208., 35.),
        232: (72., 183., 168.),
        233: (97., 99., 203.),
        242: (172., 22., 158.),
        250: (155., 64., 40.),
        261: (118., 159., 30.),
        264: (69., 252., 148.),
        276: (45., 103., 173.),
        283: (111., 38., 149.),
        286: (184., 9., 49.),
        300: (188., 174., 67.),
        304: (53., 206., 53.),
        312: (97., 235., 252.),
        323: (66., 32., 182.),
        325: (236., 114., 195.),
        331: (241., 154., 83.),
        342: (133., 240., 52.),
        356: (16., 205., 144.),
        370: (75., 101., 198.),
        392: (237., 95., 251.),
        395: (191., 52., 49.),
        399: (227., 254., 54.),
        408: (49., 206., 87.),
        417: (48., 113., 150.),
        488: (125., 73., 182.),
        540: (229., 32., 114.),
        562: (158., 119., 28.),
        570: (60., 205., 27.),
        572: (18., 215., 201.),
        581: (79., 76., 153.),
        609: (134., 13., 116.),
        748: (192., 97., 63.),
        776: (108., 163., 18.),
        1156: (95., 220., 156.),
        1163: (98., 141., 208.),
        1164: (144., 19., 193.),
        1165: (166., 36., 57.),
        1166: (212., 202., 34.),
        1167: (23., 206., 34.),
        1168: (91., 211., 236.),
        1169: (79., 55., 137.),
        1170: (182., 19., 117.),
        1171: (134., 76., 14.),
        1172: (87., 185., 28.),
        1173: (82., 224., 187.),
        1174: (92., 110., 214.),
        1175: (168., 80., 171.),
        1176: (197., 63., 51.),
        1178: (175., 199., 77.),
        1179: (62., 180., 98.),
        1180: (8., 91., 150.),
        1181: (77., 15., 130.),
        1182: (154., 65., 96.),
        1183: (197., 152., 11.),
        1184: (59., 155., 45.),
        1185: (12., 147., 145.),
        1186: (54., 35., 219.),
        1187: (210., 73., 181.),
        1188: (221., 124., 77.),
        1189: (149., 214., 66.),
        1190: (72., 185., 134.),
        1191: (42., 94., 198.),
    }
    SCANNET_COLOR_MAP_20 = {
        0: (0., 0., 0.),
        1: (174., 199., 232.),
        2: (152., 223., 138.),
        3: (31., 119., 180.),
        4: (255., 187., 120.),
        5: (188., 189., 34.),
        6: (140., 86., 75.),
        7: (255., 152., 150.),
        8: (214., 39., 40.),
        9: (197., 176., 213.),
        10: (148., 103., 189.),
        11: (196., 156., 148.),
        12: (23., 190., 207.),
        14: (247., 182., 210.),
        15: (66., 188., 102.),
        16: (219., 219., 141.),
        17: (140., 57., 197.),
        18: (202., 185., 52.),
        19: (51., 176., 203.),
        20: (200., 54., 131.),
        21: (92., 193., 61.),
        22: (78., 71., 183.),
        23: (172., 114., 82.),
        24: (255., 127., 14.),
        25: (91., 163., 138.),
        26: (153., 98., 156.),
        27: (140., 153., 101.),
        28: (158., 218., 229.),
        29: (100., 125., 154.),
        30: (178., 127., 135.),
        32: (146., 111., 194.),
        33: (44., 160., 44.),
        34: (112., 128., 144.),
        35: (96., 207., 209.),
        36: (227., 119., 194.),
        37: (213., 92., 176.),
        38: (94., 106., 211.),
        39: (82., 84., 163.),
        40: (100., 85., 144.),
    }
    if downsample is not None:
        pcd = pcd.voxel_down_sample(voxel_size=downsample)

    coord = np.asarray(pcd.points)

    # Work on kNN for coord not for color
    if scannet_data:
        label_id = []
        cls_id = []
        tree = KDTree(coord_org, leaf_size=coord_org.shape[0])

        for i in coord:
            # Get RGB
            _, match_coord = tree.query(i.reshape(1, -1), k=1)
            match_coord = match_coord[0][0]

            color_df = label_org[match_coord] * 255
            color_id = [key for key in SCANNET_COLOR_MAP_20 if np.all(SCANNET_COLOR_MAP_20[key] == color_df)]

            if len(color_id) > 0:
                cls_id.append(color_id[0])
            else:
                cls_id.append(0)

        cls_id = np.asarray(cls_id)[:, None]
        coord = coord[np.where(cls_id != 0)[0]]
        cls_id = cls_id[np.where(cls_id != 0)[0]]

        return np.hstack((cls_id, coord))
    else:
        label_id = []
        tree = KDTree(coord_org, leaf_size=coord_org.shape[0])

        for i in coord:
            _, match_coord = tree.query(i.reshape(1, -1), k=1)
            match_coord = match_coord[0][0]

            label_id.append(np.where(np.all(label_org[match_coord] == label_uniq, 1))[0][0])

        return np.hstack((np.asarray(label_id)[:, None], coord))
