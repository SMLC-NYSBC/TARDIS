"""
TARDIS - Transformer And Rapid Dimensionless Instance Segmentation

<module> Utils - Errors

New York Structural Biology Center
Simons Machine Learning Center

Robert Kiewisz, Tristan Bepler
MIT License 2021 - 2022
"""
import inspect
from os.path import expanduser, join
from typing import Tuple

from tardis.utils.logo import TardisLogo

id_dict = {
    '0': 'NO_ERROR',  # All good

    '1': 'UNKNOWN_DATA_COMPATIBILITY_ERROR',  # Data compatibility
    '100': 'DATA_COMPATIBILITY | NO_DATA_FOUND',
    '10': 'DATA_COMPATIBILITY_BUILDING_DATASET',
    '101': 'MISSING_IMAGES_OR_MASK_FILES',

    '11': 'DATA_COMPATIBILITY',  # Data set compatibility and processing.
    '111': 'WRONG_IMAGE_OR_MASK_AFTER_PREPROCESSING',
    '112': 'TRIM_SIZE_INCOMPATIBLE_WITH_ARRAY_SIZE',
    '113': 'INCORRECT_SHAPE',
    '114': 'INCORRECT_DTYPE',
    '115': 'EMPTY_OR_INCORRECT_ARRAY',
    '116': 'INCOMPATIBLE_ARRAY_AFTER_PROCESSING',

    '12': 'EMPTY_DIRECTORY',  # General
    '121': 'FILE_NOT_FOUND',
    '122': 'DIRECTORY_NOT_FOUND',
    '124': 'MISSING_OR_WRONG_PARAMETER',

    '130': 'DATA_FORMAT_INCOMPATIBLE',  # General Utils error
    '131': 'VALUE_ERROR_WHILE_LOADING_DATA',
    '132': 'NUMPY_NOT_COMPATIBLE_WITH_AMIRA',
    '133': 'VALUE_ERROR_WHILE_EXPORTING_DATA',
    '139': 'MISSING_VALUES_IN_PREDICTOR',


    '14': 'FATAL_ERROR_BUILDING_CNN',  # SpindleTorch model
    '140': 'INITIALIZATION_ERROR',
    '141': 'UNSUPPORTED_NETWORK_NAME',
    '142': 'CONVOLUTION_GROUP_NORM_ERROR',
    '143': 'CONVOLUTION_PARAMETER_VALUE_ERROR',
    '145': 'DATA_ERROR_ARRAY_SHAPE',
    '146': 'DATA_AUGMENTATION_ERROR',
    '147': 'DATA_DTYPE_ERROR',

    '19': 'AWS_INCORRECT_VALUE',  # AWS
}


def standard_error_id(id: str):
    """
    Helper function to read MRC header.

    Args:
        id (str): Tardis error id.
    """

    try:
        return id_dict[id]
    except KeyError:
        return 'UNKNOWN_ERROR'


class TardisError(Exception):
    """
    MAIN ERROR HANDLER

    Args:
        id (str): Standardized error code. See more in documentation
        py (str): .py file location
        desc (str): Error description to pass to the shell

    Returns:
        str: TARDIS Error log
    """

    def __init__(self,
                 id='0',
                 py='NA',
                 desc='Unknown exertion occurred!'):
        super().__init__()
        id_desc = standard_error_id(id)
        if id_desc == 'UNKNOWN_ERROR':
            id = 42  # Swap error for unknown error code

        self.tardis_error_rise = TardisLogo()
        prev_frame = inspect.currentframe().f_back

        dir = join(expanduser('~'), '_tardis_error.log')
        with open(dir, 'w') as f:
            f.write(f'TARDIS ERROR CODE: {id} {id_desc} \n')
            f.write(f'{prev_frame} \n')

            f.write('Location :\n')
            f.write(f'{py} \n')

            f.write('Description : \n')
            f.write(f'{desc} \n')

        desc_3, desc_4, desc_5, desc_6, \
            desc_7, desc_8, desc_9, desc_10 = self.cut_desc(desc)
        self.tardis_error_rise(title=f'TARDIS ERROR CODE {id}',
                               text_1='Error accounted in:',
                               text_2=f'{prev_frame.f_code.co_name}: {py}',
                               text_3=desc_3,
                               text_4=desc_4,
                               text_5=desc_5,
                               text_6=desc_6,
                               text_7=desc_7,
                               text_8=desc_8,
                               text_9=desc_9,
                               text_10=desc_10, )

    def cut_desc(self,
                 desc: str) -> Tuple[str, str, str, str, str, str, str, str]:
        """
        Cut string of text if too long to fit shell window.

        Args:
            desc (str): Description text.

        Returns:
            list[str]: list of cut string
        """
        WIDTH = self.tardis_error_rise.cell_width()
        if len(desc) <= WIDTH:
            text_3 = desc
            text_4, text_5, text_6, \
                text_7, text_8, text_9, text_10 = '', '', '', '', '', '', ''
        else:
            text_3, text_4, text_5, text_6, \
                text_7, text_8, text_9, text_10 = self._truncate_str(desc,
                                                                     WIDTH)

        return text_3, text_4, text_5, text_6, text_7, text_8, text_9, text_10

    @staticmethod
    def _truncate_str(desc: str,
                      width: int) -> Tuple[str, str, str, str, str, str, str, str]:
        """
        Truncate string text up to 8 string of max width fitted to shell window.
        Args:
            desc (str): Description string.
            width (int): Shell width.

        Returns:
            Tuple[str, str, str, str, str, str, str, str]: Truncate string.
        """
        MAX_TRUNC = 8
        iter_i = 0

        text = []
        while MAX_TRUNC != iter_i:
            iter_i += 1

            text.append(desc[:width])
            desc = desc[width:]

        return text
