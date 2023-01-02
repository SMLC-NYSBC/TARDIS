import inspect

from tardis.utils.logo import TardisLogo


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

        tardis_error_rise = TardisLogo()
        prev_frame = inspect.currentframe().f_back

        tardis_error_rise(title=f'TARDIS ERROR CODE {id}',
                                text_1='Error accounted in:',
                                text_2=f'{prev_frame.f_code.co_name}: {py}',
                                text_3=desc)
