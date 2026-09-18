import asyncio
import functools
import time

from reactivex.subject import Subject as rx_Subject  # re-exported for the servers

from .logging_utils import monitor_time

def timeit(name: str = "", service: str = "runtime"):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = await func(*args, **kwargs)
                end_time = time.perf_counter()
                runtime = end_time - start_time
                monitor_time(service, name or func.__name__, runtime)
                return result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                runtime = end_time - start_time
                monitor_time(service, name or func.__name__, runtime)
                return result
            return sync_wrapper
    return decorator

    
