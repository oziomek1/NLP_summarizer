import logging
import time

from typing import Any
from typing import Callable


def timeit(f: Callable) -> Callable:
    """
    Decorates function.

    :param f: Function to decorate
    :type f: Callable
    :return:
    """
    def time_calculator(*args, **kw) -> Any:
        """
        Calculates function execution time.
        :param args: Function to be measured
        :type args: callable
        :param kw: Additional parameters
        :type kw: any, optional
        :return: Function execution result
        :rtype: any
        """
        start = time.time()
        result = f(*args, **kw)
        end = time.time()

        seconds = '{:02}'.format(int(end - start))
        minutes = '{:02}:{:02}'.format(int((end - start) // 60), int((end - start) % 60))
        logging.info(f'Time spent on {args[0].logger.name} : {seconds} seconds | {minutes} minutes')
        return result
    return time_calculator
