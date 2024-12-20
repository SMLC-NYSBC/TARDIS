#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import inspect
import sys
from os.path import expanduser, join
from typing import Tuple, Union, Any

from tardis_em.utils.logo import TardisLogo

id_dict = {
    "0": "NO_ERROR",  # All good
    "01": "FEATURE_NOT_SUPPORTED_ASK_DEVELOPER",
    "1": "UNKNOWN_DATA_COMPATIBILITY_ERROR",  # Data compatibility
    "10": "DATA_COMPATIBILITY_BUILDING_DATASET",
    "100": "DATA_COMPATIBILITY | NO_DATA_FOUND",
    "101": "MISSING_IMAGES_OR_MASK_FILES",
    "11": "DATA_COMPATIBILITY",  # Data set compatibility and processing.
    "111": "WRONG_IMAGE_OR_MASK_AFTER_PREPROCESSING",
    "112": "TRIM_SIZE_INCOMPATIBLE_WITH_ARRAY_SIZE",
    "113": "INCORRECT_SHAPE",
    "114": "INCORRECT_DTYPE",
    "115": "EMPTY_OR_INCORRECT_ARRAY",
    "116": "INCOMPATIBLE_ARRAY_AFTER_PROCESSING",
    "117": "NOT_MATCHING_LENGTH_OF_LIST",
    "12": "EMPTY_DIRECTORY",  # General
    "121": "FILE_NOT_FOUND",
    "122": "DIRECTORY_NOT_FOUND",
    "124": "MISSING_OR_WRONG_PARAMETER",
    "130": "DATA_FORMAT_INCOMPATIBLE",  # General Utils error
    "131": "VALUE_ERROR_WHILE_LOADING_DATA",
    "132": "NUMPY_NOT_COMPATIBLE_WITH_AMIRA",
    "133": "VALUE_ERROR_WHILE_EXPORTING_DATA",
    "134": "VALUE_ERROR_WHILE_DRAWING_MASK",
    "139": "MISSING_VALUES_IN_PREDICTOR",
    "14": "FATAL_ERROR_BUILDING_CNN",  # SpindleTorch model
    "140": "INITIALIZATION_ERROR",
    "141": "UNSUPPORTED_NETWORK_NAME",
    "142": "CONVOLUTION_GROUP_NORM_ERROR",
    "143": "CONVOLUTION_PARAMETER_VALUE_ERROR",
    "145": "DATA_ERROR_ARRAY_SHAPE",
    "146": "DATA_AUGMENTATION_ERROR",
    "147": "DATA_DTYPE_ERROR",
    "151": "INCORRECT_PARAMETER_IN_MAIN",  # Main entry's
    "161": "MISSING_ARGUMENT",  # DIST model
    "19": "AWS_INCORRECT_VALUE",  # AWS
    "2": "MISSING_LIBRARY",
    "20": "PYTEST_ERROR",
}


def standard_error_id(id_s: str):
    """
    Fetches the standard error message corresponding to the provided ID.

    This function attempts to look up an error ID in a predefined dictionary
    and returns the associated error message. If the ID does not exist in the
    dictionary, it falls back to a default string indicating an unknown error.

    :param id_s: The identifier string for the error to search in the dictionary.
    :type id_s: str
    :return: Corresponding error message if the ID is found, else "UNKNOWN_ERROR".
    :rtype: str
    """

    try:
        return id_dict[id_s]
    except KeyError:
        return "UNKNOWN_ERROR"


class TardisError(Exception):
    """
    Handles errors specific to the TARDIS framework and provides mechanisms for
    standardized error logging, message formatting, and interactive displays.

    The class allows the creation of custom error messages with standardized identifiers,
    logs error-related information to a file, and supports formatted, error-specific console
    notifications. It also facilitates the truncation of long error descriptions to fit
    within terminal display constraints.
    """

    def __init__(
        self, id_="0", py="NA", desc="Unknown exertion occurred!", warning_b=True
    ):
        super().__init__()

        self.WIDTH = None
        id_desc = standard_error_id(id_)
        if id_desc == "UNKNOWN_ERROR":
            id_ = 42  # Swap error for unknown error code

        self.tardis_error_rise = TardisLogo()
        prev_frame = inspect.currentframe().f_back

        dir_ = join(expanduser("~"), "_tardis_error.log")
        with open(dir_, "w") as f:
            f.write(f"TARDIS ERROR CODE: {id_} {id_desc} \n")
            f.write(f"{prev_frame} \n")
            f.write("\n")
            f.write("Location :\n")
            f.write(f"{py} \n")
            f.write("\n")
            f.write("Description : \n")
            f.write(f"{desc} \n")

        desc_3, desc_4, desc_5, desc_6, desc_7, desc_8, desc_9, desc_10 = self.cut_desc(
            desc
        )
        code = "ERROR" if not warning_b else "WARNING"

        self.tardis_error_rise(
            title=f"TARDIS {code} CODE {id_} {id_desc} \n",
            text_1="Error accounted in:",
            text_2=f"{prev_frame.f_code.co_name}: {py}",
            text_3=desc_3,
            text_4=desc_4,
            text_5=desc_5,
            text_6=desc_6,
            text_7=desc_7,
            text_8=desc_8,
            text_9=desc_9,
            text_10=desc_10,
        )
        if not warning_b:
            sys.exit()

    def cut_desc(self, desc: str) -> Tuple[str, str, str, str, str, str, str, str]:
        """
        Splits the given string `desc` into multiple smaller strings, ensuring each
        split string does not exceed the defined width. The method first retrieves
        the available width for truncation and then decides whether the input string
        needs splitting. If the string is shorter than or equal to the width, no
        splitting is done; otherwise, the string is truncated into multiple parts
        using the `_truncate_str` method.

        :param desc: The input string to be truncated and split into smaller parts.
        :type desc: str
        :return: A tuple containing up to eight parts of the truncated string.
            Parts that are not used remain as empty strings.
        :rtype: Tuple[str, str, str, str, str, str, str, str]
        """
        self.tardis_error_rise.cell_width()

        WIDTH = self.tardis_error_rise.WIDTH - 21
        if len(desc) <= WIDTH:
            text_3 = desc
            text_4, text_5, text_6, text_7, text_8, text_9, text_10 = (
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            )
        else:
            (
                text_3,
                text_4,
                text_5,
                text_6,
                text_7,
                text_8,
                text_9,
                text_10,
            ) = self._truncate_str(desc, WIDTH)

        return text_3, text_4, text_5, text_6, text_7, text_8, text_9, text_10

    @staticmethod
    def _truncate_str(desc: str, width: int) -> Union[str, Any]:
        """
        Static method that truncates the given string into segments of a specified width.
        It iteratively processes the string and splits it into chunks of a defined
        width, limiting the total number of chunks to a maximum predefined count.

        :param desc: Input string to be truncated.
        :type desc: str
        :param width: Width of each truncated segment.
        :type width: int
        :return: A list containing truncated segments of the string, or any fallback
            result in case of unexpected conditions.
        :rtype: Union[str, Any]
        """
        MAX_TRUNC = 8
        iter_i = 0

        text_ = []
        while MAX_TRUNC != iter_i:
            iter_i += 1

            text_.append(desc[:width])
            desc = desc[width:]

        return text_
