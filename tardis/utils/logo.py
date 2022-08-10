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
         f' |TARDIS {version}    {title}\n'
          ' |New York Structural Biology Center                        ___        |\n'
          ' |Simons Machine Learning Center                    _______(_@_)_______|\n'
          ' |                                                  | POLICE      BOX ||\n'
          ' |                                                  |_________________||\n'
          ' |                                                   | _____ | _____ | |\n'
          ' |                                                   | |###| | |###| | |\n'
          ' |                                                   | _____ | _____ | |\n'
          ' |                                                   | ||_|| | ||_|| | |\n'
          ' |                                                   | _____ |$_____ | |\n'
          ' |                                                   | || || | || || | |\n'
          ' |                                                   | _____ | _____ | |\n'
          ' | TARDIS-pytorch Copyright Information:             | || || | || || | |\n'
          ' | Copyright (c) 2021 Robert Kiewisz, Tristan Bepler |       |       | |\n'
          ' | MIT License                                       ***************** |\n'
          '  =====================================================================\n')
