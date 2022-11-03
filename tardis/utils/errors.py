from tardis.utils.logo import Tardis_Logo
import inspect


class TardisError(Exception):
    def __init__(self, ID='0', py='NA', desc='Unknown exertion occurred!'):
        super().__init__()
        tardis_error_rise = Tardis_Logo()
        prev_frame = inspect.currentframe().f_back

        tardis_error_rise(title=f'TARDIS ERROR CODE {ID}',
                          text_1=f'Error accounter in:',
                          text_2=f'{prev_frame.f_code.co_name}: {py}',
                          text_3=desc)
