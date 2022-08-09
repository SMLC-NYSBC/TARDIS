from os import system
import platform
from tardis.version import version


def tardis_logo(title: str):
    if platform.system() in ['Darwin', 'Linux']:
        clear = lambda: system('clear')
    else:
        clear = lambda: system('cls')

    clear()
    print('=====================================================================\n')
    print(f'TARDIS {version}')
    print(f'{title}')
    print('New York Structural Biology Center - Simons Machine Learning Center\n')
    print('TARDIS-pytorch Copyright Information:\n')
    print('Copyright (c) 2021 Robert Kiewisz, Tristan Bepler')
    print('MIT License\n')
    print('=====================================================================\n')