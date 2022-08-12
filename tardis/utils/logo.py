from os import system
import platform
from tardis.version import version
import sys


def printProgressBar(value, max):
    n_bar = 40  # size of progress bar
    j = value / max
    bar = '█' * int(n_bar * j)
    bar = bar + '-' * int(n_bar * (1 - j))

    return f"[{bar:{n_bar}s}] {int(100 * j)}%"


def build_text(text=''):
    max = 49
    text_len = len(text)
    if text_len > max:
        return text[:max]
    else:
        new_len = max - text_len
        return f'{text +  " " * new_len}'


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def clear_output(wait=True):
    """Clear the output of the current cell receiving output.

    Parameters
    ----------
    wait : bool [default: false]
        Wait to clear the output until new output is available to replace it."""
    from IPython.core.interactiveshell import InteractiveShell
    if InteractiveShell.initialized():
        InteractiveShell.instance().display_pub.clear_output(wait)
    else:
        print('\033[2K\r', end='')
        sys.stdout.flush()
        print('\033[2K\r', end='')
        sys.stderr.flush()


class Tardis_Logo:
    def __init__(self):
        clear = None

        if platform.system() in ['Darwin', 'Linux']:
            self.clear = lambda: system('clear')
        else:
            self.clear = lambda: system('cls')

        if clear is None and is_interactive():
            self.clear = clear_output

    def __call__(self,
                 title: str, text_1='', text_2='', text_3='', text_4='',
                 text_5='', text_6='', text_7=''):
        self.clear()
        print('  =====================================================================\n'
              f' | TARDIS {build_text(version + "  " + title)}            |\n'
              ' |                                                                     |\n'
              ' | New York Structural Biology Center                       ___        |\n'
              ' | Simons Machine Learning Center                   _______(_@_)_______|\n'
              ' |                                                  | POLICE      BOX ||\n'
              f' | {build_text(text_1)}|_________________||\n'
              f' | {build_text(text_2)} | _____ | _____ | |\n'
              f' | {build_text(text_3)} | |###| | |###| | |\n'
              f' | {build_text(text_4)} | _____ | _____ | |\n'
              f' | {build_text(text_5)} | ||_|| | ||_|| | |\n'
              f' | {build_text(text_6)} | _____ |$_____ | |\n'
              f' | {build_text(text_7)} | || || | || || | |\n'
              ' |                                                   | _____ | _____ | |\n'
              ' | TARDIS-pytorch Copyright Information:             | || || | || || | |\n'
              ' | Copyright (c) 2021 Robert Kiewisz, Tristan Bepler |       |       | |\n'
              ' | MIT License                                       ***************** |\n'
              '  =====================================================================\n')
