import struct
from collections import namedtuple

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}

# int nx
# int ny
# int nz
fstr = "3i"
names = "nx ny nz"

# int mode
fstr += "i"
names += " mode"

# int nxstart
# int nystart
# int nzstart
fstr += "3i"
names += " nxstart nystart nzstart"

# int mx
# int my
# int mz
fstr += "3i"
names += " mx my mz"

# float xlen
# float ylen
# float zlen
fstr += "3f"
names += " xlen ylen zlen"

# float alpha
# float beta
# float gamma
fstr += "3f"
names += " alpha beta gamma"

# int mapc
# int mapr
# int maps
fstr += "3i"
names += " mapc mapr maps"

# float amin
# float amax
# float amean
fstr += "3f"
names += " amin amax amean"

# int ispg
# int next
# short creatid
fstr += "2ih"
names += " ispg next creatid"

# pad 30 (extra data)
# [98:128]
fstr += "30x"

# short nint
# short nreal
fstr += "2h"
names += " nint nreal"

# pad 20 (extra data)
# [132:152]
fstr += "20x"

# int imodStamp
# int imodFlags
fstr += "2i"
names += " imodStamp imodFlags"

# short idtype
# short lens
# short nd1
# short nd2
# short vd1
# short vd2
fstr += "6h"
names += " idtype lens nd1 nd2 vd1 vd2"

# float[6] tiltangles
fstr += "6f"
names += " tilt_ox tilt_oy tilt_oz tilt_cx tilt_cy tilt_cz"

# NEW-STYLE MRC image2000 HEADER - IMOD 2.6.20 and above
# float xorg
# float yorg
# float zorg
# char[4] cmap
# char[4] stamp
# float rms
fstr += "3f4s4sf"
names += " xorg yorg zorg cmap stamp rms"

# int nlabl
# char[10][80] labels
fstr += "i800s"
names += " nlabl labels"

header_struct = struct.Struct(fstr)
MRCHeader = namedtuple("MRCHeader", names)

rgb_color = {
    "black": (0, 0, 0),
    "white": (1, 1, 1),
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "yellow": (1, 1, 0),
    "cyan": (0, 1, 1),
    "magenta": (1, 0, 1),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
    "light_gray": (0.75, 0.75, 0.75),
    "light_grey": (0.75, 0.75, 0.75),
    "dark_gray": (0.25, 0.25, 0.25),
    "dark_grey": (0.25, 0.25, 0.25),
    "orange": (1, 0.65, 0),
    "purple": (0.5, 0, 0.5),
    "pink": (1, 0.75, 0.8),
    "brown": (0.6, 0.4, 0.2),
    "lime": (0.75, 1, 0),
    "navy": (0, 0, 0.5),
    "olive": (0.5, 0.5, 0),
    "teal": (0, 0.5, 0.5),
}
