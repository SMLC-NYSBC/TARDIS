import platform
import sys
from os import get_terminal_size, system

from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output
from tardis.version import version


def printProgressBar(value: int,
                     max: int):
    """
    Builder for ASCII type progress bar.

    Args:
        value (int): Current value for the progress bar.
        max (int): Maximum number of iterations.
    """
    if is_interactive():
        n_bar = 75
    else:
        try:
            n_bar = get_terminal_size()[0] - 50
        except OSError:
            n_bar = 50

    j = value / max
    bar = '█' * int(n_bar * j)
    bar = bar + '-' * int(n_bar * (1 - j))

    return f"[{bar:{n_bar}s}] {int(100 * j)}%  [{value} / {max}]"


def is_interactive():
    """
    Simple check if command line window is from Jupiter.
    """
    import __main__ as main

    return not hasattr(main, '__file__')


class Tardis_Logo:
    """
    BUILDER FOR LOG OUTPUT

    The log output is built to fit into the given window (cmd or Jupyter)
    The side Logo is optional and can be removed upon; set the logo to False
    The title and text are optional.

    Example:

    log = Tardis_Logo()

    log(title='Example',

    text_1='Progress bar:',

    text_2=printProgressBar(value=i, max=len(range(10))),

    ....)
    """

    def __init__(self):
        self.CLEAR  = None

        if platform.system() in ['Darwin', 'Linux']:
            self.CLEAR = lambda: system('clear')
        else:
            self.CLEAR = lambda: system('cls')

        if is_interactive():
            self.CLEAR = lambda: clear_output(wait=True)

        self.FN = "TARDIS-pytorch Copyright Information:"
        self.C = "Copyright (c) 2021 Robert Kiewisz, Tristan Bepler"

    @staticmethod
    def _build_text(max=80,
                    text='',
                    repeat=False) -> str:
        """
        Format text input to fit the pre-define window size.

        Args:
            max (int): Width of the console window. Defaults to 80.
            text (str): Text value to be included in the output.
            repeat (bool): If True repeat text input till the end of max value.

        Returns:
            str: Formatted text string with max number of characters.
        """
        max = max
        if repeat:
            text = text * max
        text_len = len(text)

        if text_len > max:
            return text[:max]
        else:
            new_len = max - text_len
            return f'{text +  " " * new_len}'

    @staticmethod
    def clear_output(wait=True):
        """
        Clear window output other Jupyter of the command line window.

        Args:
            wait (bool):  Wait to clear the output until the new output is
                available to replace it.
        """
        if InteractiveShell.initialized():
            InteractiveShell.instance().display_pub.clear_output(wait)
        else:
            print('\033[2K\r', end='')
            sys.stdout.flush()
            print('\033[2K\r', end='')
            sys.stderr.flush()

    def __call__(self,
                 title='', text_1='', text_2='', text_3='', text_4='',
                 text_5='', text_6='', text_7='', text_8='', text_9='', text_10='',
                 logo=True):
        """
        Builder call function to output nice looking progress bar.

        Args:
            title (str, optional): Any text string.
            text_1 (str, optional): Any text string.
            text_2 (str, optional): Any text string.
            text_3 (str, optional): Any text string.
            text_4 (str, optional): Any text string.
            text_5 (str, optional): Any text string.
            text_6 (str, optional): Any text string.
            text_7 (str, optional): Any text string.
            text_8 (str, optional): Any text string.
            text_9 (str, optional): Any text string.
            text_10 (str, optional): Any text string.
            logo (bool, optional): Any text string.

        Returns:
            print: Print progress bar with all text options into cmd commend line
                window or jupyter notebook.
        """
        # Check and update window size
        if is_interactive():
            WIDTH = 75
        else:
            try:
                WIDTH = get_terminal_size()[0] - 5
            except OSError:
                WIDTH = 50

        self.CLEAR()
        if logo:
            print(f'  {self._build_text(WIDTH + 1, "=", True)}\n'
                  f' | {self._build_text(WIDTH, "TARDIS  " + version + "  " + title)}|\n'
                  f' | {self._build_text(WIDTH, " ", True)}|\n'
                  f' | {self._build_text(WIDTH, "New York Structural Biology Center")}|\n'
                  f' | {self._build_text(WIDTH - 13, "Simons Machine Learning Center")} ___         |\n'
                  f' | {self._build_text(WIDTH - 21, " ", True)} _______(_@_)_______ |\n'
                  f' | {self._build_text(WIDTH - 21, " ", True)} |  TARDIS-pytorch | |\n'
                  f' | {self._build_text(WIDTH - 21, text_1)} |_________________| |\n'
                  f' | {self._build_text(WIDTH - 21, text_2)}  | _____ | _____ |  |\n'
                  f' | {self._build_text(WIDTH - 21, text_3)}  | |###| | |###| |  |\n'
                  f' | {self._build_text(WIDTH - 21, text_4)}  | |###| | |###| |  |\n'
                  f' | {self._build_text(WIDTH - 21, text_5)}  | _____ | _____ |  |\n'
                  f' | {self._build_text(WIDTH - 21, text_6)}  | || || | || || |  |\n'
                  f' | {self._build_text(WIDTH - 21, text_7)}  | ||_|| | ||_|| |  |\n'
                  f' | {self._build_text(WIDTH - 21, text_8)}  | _____ |$_____ |  |\n'
                  f' | {self._build_text(WIDTH - 21, text_9)}  | || || | || || |  |\n'
                  f' | {self._build_text(WIDTH - 21, text_10)}  | ||_|| | ||_|| |  |\n'
                  f' | {self._build_text(WIDTH - 21, " ",  True)}  | _____ | _____ |  |\n'
                  f' | {self._build_text(WIDTH - 21, " ",  True)}  | || || | || || |  |\n'
                  f' | {self._build_text(WIDTH - 21, self.FN)}  | ||_|| | ||_|| |  |\n'
                  f' | {self._build_text(WIDTH - 21, self.C)}  |       |       |  |\n'
                  f' | {self._build_text(WIDTH - 21, "MIT License")}  *****************  |\n'
                  f'  {self._build_text(WIDTH + 1, "=", True)}\n')
        else:
            print(f'  {self._build_text(WIDTH + 1, "=", True)}\n'
                  f' | {self._build_text(WIDTH, "TARDIS  " + version + "  " + title)}|\n'
                  f' | {self._build_text(WIDTH, " ", True)}|\n'
                  f' | {self._build_text(WIDTH, "New York Structural Biology Center")}|\n'
                  f' | {self._build_text(WIDTH, "Simons Machine Learning Center")}|\n'
                  f' | {self._build_text(WIDTH, " ", True)}|\n'
                  f' | {self._build_text(WIDTH, " ", True)}|\n'
                  f' | {self._build_text(WIDTH, text_1)}|\n'
                  f' | {self._build_text(WIDTH, text_2)}|\n'
                  f' | {self._build_text(WIDTH, text_3)}|\n'
                  f' | {self._build_text(WIDTH, text_4)}|\n'
                  f' | {self._build_text(WIDTH, text_5)}|\n'
                  f' | {self._build_text(WIDTH, text_6)}|\n'
                  f' | {self._build_text(WIDTH, text_7)}|\n'
                  f' | {self._build_text(WIDTH, text_8)}|\n'
                  f' | {self._build_text(WIDTH, text_9)}|\n'
                  f' | {self._build_text(WIDTH, text_10)}|\n'
                  f' | {self._build_text(WIDTH, " ",  True)}|\n'
                  f' | {self._build_text(WIDTH, " ",  True)}|\n'
                  f' | {self._build_text(WIDTH, self.FN)}|\n'
                  f' | {self._build_text(WIDTH, self.C)}|\n'
                  f' | {self._build_text(WIDTH, "MIT License")}|\n'
                  f'  {self._build_text(WIDTH + 1, "=", True)}\n')