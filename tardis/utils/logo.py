import platform
import sys
from os import get_terminal_size, system

from IPython.core.interactiveshell import InteractiveShell
from tardis.version import version


def printProgressBar(value: int,
                     max: int):
    """
    BUILDER FOR ASCII TYPE PROGRESS BAR

        Args:
            value: Current value for the progress bar
            max: Maximum number of iterations
    """
    if is_interactive():
        n_bar = 75
    else:
        n_bar = get_terminal_size()[0] - 50

    j = value / max
    bar = '█' * int(n_bar * j)
    bar = bar + '-' * int(n_bar * (1 - j))

    return f"[{bar:{n_bar}s}] {int(100 * j)}%"


def build_text(max=80,
               text='',
               repeat=False):
    """
    BUILD TEXT WITH UNIFIED STYLE FOR LOG WINDOW

    Args:
        max: Width of console window. Defaults to 80.
        text: Text value to be included in the output. Defaults to ''.
        repeat: If True repeat text input till end of max value. Defaults to False.
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


def is_interactive():
    """
    CHECK IF JUPYTER
    """
    import __main__ as main
    return not hasattr(main, '__file__')


def clear_output(wait=True):
    """
    CLEAR THE OUTPUT OF THE WINDOW

    Args:
        wait: Wait to clear the output until new output is available to replace it.
    """
    if InteractiveShell.initialized():
        InteractiveShell.instance().display_pub.clear_output(wait)
    else:
        print('\033[2K\r', end='')
        sys.stdout.flush()
        print('\033[2K\r', end='')
        sys.stderr.flush()


class Tardis_Logo:
    """
    BUILDER FOR LOG OUTPUT 

    ============= The log output is build to fit into given window (cmd or Jupyter)
    |           | Side Logo is optional and can be removed upon; set logo to False
    |  Example  | Title and text are optional
    |           | Ex. log = Tardis_Logo()
    =============     log(title='Example',
                          title_1='Progress bar:',
                          title_2=printProgressBar(value=i, max=len(range(10))))
    """
    def __init__(self):
        clear = None

        if platform.system() in ['Darwin', 'Linux']:
            self.clear = lambda: system('clear')
        else:
            self.clear = lambda: system('cls')

        if clear is None and is_interactive():
            self.clear = clear_output

    def __call__(self,
                 title='', text_1='', text_2='', text_3='', text_4='',
                 text_5='', text_6='', text_7='', text_8='', text_9='', text_10='',
                 logo=True):
        if is_interactive():
            max_width = 75
        else:
            max_width = get_terminal_size()[0] - 5

        self.clear()
        if logo:
            print(f'  {build_text(max_width + 1, "=", True)}\n'
                f' | {build_text(max_width, "TARDIS  " + version + "  " + title)}|\n'
                f' | {build_text(max_width, " ", True)}|\n'
                f' | {build_text(max_width, "New York Structural Biology Center")}|\n'
                f' | {build_text(max_width - 13, "Simons Machine Learning Center")} ___         |\n'
                f' | {build_text(max_width - 21, " ", True)} _______(_@_)_______ |\n'
                f' | {build_text(max_width - 21, " ", True)} | POLICE      BOX | |\n'
                f' | {build_text(max_width - 21, text_1)} |_________________| |\n'
                f' | {build_text(max_width - 21, text_2)}  | _____ | _____ |  |\n'
                f' | {build_text(max_width - 21, text_3)}  | |###| | |###| |  |\n'
                f' | {build_text(max_width - 21, text_4)}  | |###| | |###| |  |\n'
                f' | {build_text(max_width - 21, text_5)}  | _____ | _____ |  |\n'
                f' | {build_text(max_width - 21, text_6)}  | || || | || || |  |\n'
                f' | {build_text(max_width - 21, text_7)}  | ||_|| | ||_|| |  |\n'
                f' | {build_text(max_width - 21, text_8)}  | _____ |$_____ |  |\n'
                f' | {build_text(max_width - 21, text_9)}  | || || | || || |  |\n'
                f' | {build_text(max_width - 21, text_10)}  | ||_|| | ||_|| |  |\n'
                f' | {build_text(max_width - 21, " ",  True)}  | _____ | _____ |  |\n'
                f' | {build_text(max_width - 21, " ",  True)}  | || || | || || |  |\n'
                f' | {build_text(max_width - 21, "TARDIS-pytorch Copyright Information:")}  | ||_|| | ||_|| |  |\n'
                f' | {build_text(max_width - 21, "Copyright (c) 2021 Robert Kiewisz, Tristan Bepler")}  |       |       |  |\n'
                f' | {build_text(max_width - 21, "MIT License")}  *****************  |\n'
                f'  {build_text(max_width + 1, "=", True)}\n')
        else:
            print(f'  {build_text(max_width + 1, "=", True)}\n'
                f' | {build_text(max_width, "TARDIS  " + version + "  " + title)}|\n'
                f' | {build_text(max_width, " ", True)}|\n'
                f' | {build_text(max_width, "New York Structural Biology Center")}|\n'
                f' | {build_text(max_width, "Simons Machine Learning Center")}|\n'
                f' | {build_text(max_width, " ", True)}|\n'
                f' | {build_text(max_width, " ", True)}|\n'
                f' | {build_text(max_width, text_1)}|\n'
                f' | {build_text(max_width, text_2)}|\n'
                f' | {build_text(max_width, text_3)}|\n'
                f' | {build_text(max_width, text_4)}|\n'
                f' | {build_text(max_width, text_5)}|\n'
                f' | {build_text(max_width, text_6)}|\n'
                f' | {build_text(max_width, text_7)}|\n'
                f' | {build_text(max_width, text_8)}|\n'
                f' | {build_text(max_width, text_9)}|\n'
                f' | {build_text(max_width, text_10)}|\n'
                f' | {build_text(max_width, " ",  True)}|\n'
                f' | {build_text(max_width, " ",  True)}|\n'
                f' | {build_text(max_width, "TARDIS-pytorch Copyright Information:")}|\n'
                f' | {build_text(max_width, "Copyright (c) 2021 Robert Kiewisz, Tristan Bepler")}|\n'
                f' | {build_text(max_width, "MIT License")}|\n'
                f'  {build_text(max_width + 1, "=", True)}\n')
