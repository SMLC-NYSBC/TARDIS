#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
import logging
import platform
import socket
import sys
from os import get_terminal_size, system

from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output

from tardis_em._version import version


def print_progress_bar(value: int, max_v: int):
    """
    Displays a progress bar indicating the percentage of completion. The progress
    bar is dynamically adjusted based on the terminal's width in interactive
    mode. In non-interactive environments or in case of error, a default width
    is used.

    :param value: The current progress value. Must be a non-negative integer
                  less than or equal to `max_v`.
    :param max_v: The maximum value for the progress bar, representing 100%
                  progress. Must be a positive integer greater than 0.

    :return: A formatted string representation of a progress bar including the
             current percentage of completion and value status.
    :rtype: str
    """
    if is_interactive():
        n_bar = 50
    else:
        try:
            n_bar = get_terminal_size()[0] - 50
        except OSError:
            n_bar = 50

    j = value / max_v
    bar = "█" * int(n_bar * j)
    bar = bar + "-" * int(n_bar * (1 - j))

    return f"[{bar:{n_bar}s}] {int(100 * j)}%  [{value} / {max_v}]"


def is_interactive():
    """
    Simple check if a command line window is from Jupiter.
    """
    import __main__ as main

    return not hasattr(main, "__file__")


class TardisLogo:
    """
    Class for managing TARDIS logo display and formatting within a console or
    Jupyter environment.
    """

    def __init__(self, logo=True):
        self.title = ""

        self.CLEAR = None
        self.WIDTH = None

        if platform.system() in ["Darwin", "Linux"]:
            self.CLEAR = lambda: print("\x1b[2J\x1b[H")
        else:
            self.CLEAR = lambda: system("cls")

        if is_interactive():
            self.CLEAR = lambda: clear_output(wait=True)

        self.FN = "TARDIS-pytorch Copyright Information:"
        self.C = "Copyright (c) 2021 Robert Kiewisz, Tristan Bepler"

        self.logo = logo

    @staticmethod
    def _build_text(max_i=80, text="", repeat=False) -> str:
        """
        Constructs a formatted text string based on the maximum length,
        given text, and repeat flag.

        :param max_i: Maximum length of the resulting string.
        :type max_i: int
        :param text: Input text to format.
        :type text: str
        :param repeat: Flag indicating whether the input text should
            be repeated to fill the maximum length.
        :type repeat: bool

        :return: A formatted text string of exact length `max_`. The
            string is either truncated or padded with spaces as needed.
        :rtype: str
        """
        if repeat:
            text = text * max_i

        return f'{text[:max_i] + " " * (max_i - len(text))}'

    @staticmethod
    def clear_output(wait=True):
        """
        Clears the current output displayed in the terminal or within an interactive
        environment.

        :param wait: Determines whether the clearing of the output should wait
            for the display to be updated. The functionality of this parameter
            applies specifically when an interactive shell is being used.
        :type wait: bool
        """
        if InteractiveShell.initialized():
            InteractiveShell.instance().display_pub.clear_output(wait)
        else:
            print("\033[2K\r", end="")
            sys.stdout.flush()
            print("\033[2K\r", end="")
            sys.stderr.flush()

    def cell_width(self):
        """
        Adjusts the cell width based on the environment and terminal properties.
        """
        if is_interactive():  # Jupyter
            self.WIDTH = 90
        else:
            try:  # Console
                self.WIDTH = get_terminal_size()[0] - 5
            except OSError:  # Any other
                self.WIDTH = 100

    def __call__(
        self,
        title="",
        text_0=" ",
        text_1="",
        text_2="",
        text_3="",
        text_4="",
        text_5="",
        text_6="",
        text_7="",
        text_8="",
        text_9="",
        text_10="",
        text_11=" ",
    ):
        """
        Executes the callable logic for rendering a visual text representation of a logo
        or textual information depending on the configuration. It adjusts variables such
        as title and checks window size compatibility before displaying the output. The
        logo and textual components are formatted based on predefined templates, with
        customizable fields like texts and titles.

        :param title: Title to display on top of the text or logo, defaults to an empty string.
        :type title: str
        :param text_0: Customizable text field number 0, defaults to an empty string with a space.
        :type text_0: str
        :param text_1: Customizable text field number 1, defaults to an empty string.
        :type text_1: str
        :param text_2: Customizable text field number 2, defaults to an empty string.
        :type text_2: str
        :param text_3: Customizable text field number 3, defaults to an empty string.
        :type text_3: str
        :param text_4: Customizable text field number 4, defaults to an empty string.
        :type text_4: str
        :param text_5: Customizable text field number 5, defaults to an empty string.
        :type text_5: str
        :param text_6: Customizable text field number 6, defaults to an empty string.
        :type text_6: str
        :param text_7: Customizable text field number 7, defaults to an empty string.
        :type text_7: str
        :param text_8: Customizable text field number 8, defaults to an empty string.
        :type text_8: str
        :param text_9: Customizable text field number 9, defaults to an empty string.
        :type text_9: str
        :param text_10: Customizable text field number 10, defaults to an empty string.
        :type text_10: str
        :param text_11: Customizable text field number 11, defaults to an empty string with a space.
        :type text_11: str
        :return: None
        """
        if title != "":
            self.title = title

        # Check and update window size
        self.cell_width()

        self.CLEAR()
        if self.logo:
            logo = (
                f'  {self._build_text(self.WIDTH + 1, "=", True)}\n'
                + f' | {self._build_text(self.WIDTH, "TARDIS  " + version + "  " + self.title)}|\n'
                + f' | {self._build_text(self.WIDTH, " ", True)}|\n'
                + f' | {self._build_text(self.WIDTH, "New York Structural Biology Center")}|\n'
                + f' | {self._build_text(self.WIDTH - 13, "Simons Machine Learning Center")} ___         |\n'
                + f' | {self._build_text(self.WIDTH - 21, " ", True)} _______(_@_)_______ |\n'
                + f" | {self._build_text(self.WIDTH - 21, text_0)} |     TARDIS-em   | |\n"
                + f" | {self._build_text(self.WIDTH - 21, text_1)} |_________________| |\n"
                + f" | {self._build_text(self.WIDTH - 21, text_2)}  | _____ | _____ |  |\n"
                + f" | {self._build_text(self.WIDTH - 21, text_3)}  | |###| | |###| |  |\n"
                + f" | {self._build_text(self.WIDTH - 21, text_4)}  | |###| | |###| |  |\n"
                + f" | {self._build_text(self.WIDTH - 21, text_5)}  | _____ | _____ |  |\n"
                + f" | {self._build_text(self.WIDTH - 21, text_6)}  | || || | || || |  |\n"
                + f" | {self._build_text(self.WIDTH - 21, text_7)}  | ||_|| | ||_|| |  |\n"
                + f" | {self._build_text(self.WIDTH - 21, text_8)}  | _____ |$_____ |  |\n"
                + f" | {self._build_text(self.WIDTH - 21, text_9)}  | || || | || || |  |\n"
                + f" | {self._build_text(self.WIDTH - 21, text_10)}  | ||_|| | ||_|| |  |\n"
                + f" | {self._build_text(self.WIDTH - 21, text_11)}  | _____ | _____ |  |\n"
                + f' | {self._build_text(self.WIDTH - 21, " ", True)}  | || || | || || |  |\n'
                + f" | {self._build_text(self.WIDTH - 21, self.FN)}  | ||_|| | ||_|| |  |\n"
                + f" | {self._build_text(self.WIDTH - 21, self.C)}  |       |       |  |\n"
                + f' | {self._build_text(self.WIDTH - 21, "MIT License")}  *****************  |\n'
                + f'  {self._build_text(self.WIDTH + 1, "=", True)}\n'
            )
            print(logo)
        else:
            logo = (
                f'  {self._build_text(self.WIDTH + 1, "=", True)}\n'
                + f' | {self._build_text(self.WIDTH, "TARDIS  " + version + "  " + self.title)}|\n'
                + f' | {self._build_text(self.WIDTH, " ", True)}|\n'
                + f' | {self._build_text(self.WIDTH, "New York Structural Biology Center")}|\n'
                + f' | {self._build_text(self.WIDTH, "Simons Machine Learning Center")}|\n'
                + f' | {self._build_text(self.WIDTH, " ", True)}|\n'
                + f" | {self._build_text(self.WIDTH, text_0)}|\n"
                + f" | {self._build_text(self.WIDTH, text_1)}|\n"
                + f" | {self._build_text(self.WIDTH, text_2)}|\n"
                + f" | {self._build_text(self.WIDTH, text_3)}|\n"
                + f" | {self._build_text(self.WIDTH, text_4)}|\n"
                + f" | {self._build_text(self.WIDTH, text_5)}|\n"
                + f" | {self._build_text(self.WIDTH, text_6)}|\n"
                + f" | {self._build_text(self.WIDTH, text_7)}|\n"
                + f" | {self._build_text(self.WIDTH, text_8)}|\n"
                + f" | {self._build_text(self.WIDTH, text_9)}|\n"
                + f" | {self._build_text(self.WIDTH, text_10)}|\n"
                + f" | {self._build_text(self.WIDTH, text_11)}|\n"
                + f' | {self._build_text(self.WIDTH, " ", True)}|\n'
                + f" | {self._build_text(self.WIDTH, self.FN)}|\n"
                + f" | {self._build_text(self.WIDTH, self.C)}|\n"
                + f' | {self._build_text(self.WIDTH, "MIT License")}|\n'
                + f'  {self._build_text(self.WIDTH + 1, "=", True)}\n'
            )
            print(logo)


class ContextFilter(logging.Filter):
    hostname = socket.gethostname()

    def filter(self, record):
        record.hostname = ContextFilter.hostname
        return True
