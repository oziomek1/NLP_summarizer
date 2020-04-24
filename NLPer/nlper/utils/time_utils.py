import logging
import time


def timeit(f):
    def time_calculator(*args, **kw):
        start = time.time()
        result = f(*args, **kw)
        end = time.time()

        seconds = '{:02}'.format(int(end - start))
        minutes = '{:02}:{:02}'.format(int((end - start) // 60), int((end - start) % 60))
        logging.info(f'Time spent on {args[0].logger.name} : {seconds} seconds | {minutes} minutes')
        return result
    return time_calculator
