from os import system
import platform
from tardis.version import version


def tardis_logo(title: str):
    if platform.system() in ['Darwin', 'Linux']:
        clear = lambda: system('clear')
    else:
        clear = lambda: system('cls')

    clear()

    print('  =====================================================================\n'
         f' | TARDIS {build_text(version + "  " + title)}            |\n'
          ' | New York Structural Biology Center                       ___        |\n'
          ' | Simons Machine Learning Center                   _______(_@_)_______|\n'
          ' |                                                  | POLICE      BOX ||\n'
         f' | {build_text()}|_________________||\n'
         f' | {build_text()} | _____ | _____ | |\n'
         f' | {build_text()} | |###| | |###| | |\n'
         f' | {build_text()} | _____ | _____ | |\n'
         f' | {build_text()} | ||_|| | ||_|| | |\n'
         f' | {build_text()} | _____ |$_____ | |\n'
         f' | {build_text()} | || || | || || | |\n'
          ' |                                                   | _____ | _____ | |\n'
          ' | TARDIS-pytorch Copyright Information:             | || || | || || | |\n'
          ' | Copyright (c) 2021 Robert Kiewisz, Tristan Bepler |       |       | |\n'
          ' | MIT License                                       ***************** |\n'
          '  =====================================================================\n')

def build_text(text=''):
    max = 49
    text_len = len(text)
    if text > max:
        return text[:max]
    else:
        new_len = max - text_len
        return f'{text +  " " * new_len}'