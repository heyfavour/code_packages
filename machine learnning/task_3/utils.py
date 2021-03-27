import datetime
from functools import wraps


def func_log(func):
    @wraps(func)
    def warpper(*args, **kwargs):
        start = datetime.datetime.now()
        func_result = func(*args, **kwargs)
        end = datetime.datetime.now()
        delta = end - start
        print(f"{func.__name__}:{delta}")
        return func_result
    return warpper
