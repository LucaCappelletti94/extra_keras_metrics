from decorator import decorator
from typing import Callable, List, Dict
from .metric import metric
import tensorflow as tf
import math
import numpy as np

def format_int(param:int):
    return str(param)

def format_float(param:float):
    mantissa, integer= math.modf(param)
    return "{integer}_{mantissa}".format(
        integer=int(integer),
        mantissa=int(mantissa*10000)
    ).strip("0")

def format_list(param:List):
    return "-".join(
        format_parameter(p) for p in param
    )

def format_parameter(param):
    if isinstance(param, int):
        return format_int(param)
    if isinstance(param, float):
        return format_float(param)
    if isinstance(param, list):
        return format_list(param)
    if isinstance(param, tf.Tensor):
        return ""
    raise ValueError("Unknown type!")

def format_args(args:List):
    if args:
        return "_{args}".format(
            args="_".join([
                format_parameter(arg) for arg in args
            ])
        )
    return ""

def format_kwargs(kwargs:Dict):
    if kwargs:
        return "_{kwargs}".format(
            kwargs="_".join([
                "{key}-{arg}".format(
                    key=key,
                    arg=format_parameter(arg)
                ) for key, arg in kwargs.items()
            ])
        )
    return ""

@decorator
def parametrized_metric(parametrized_metric:Callable[..., Callable[[tf.Tensor, tf.Tensor], float]], *args:List, **kwargs:Dict):
    metric = parametrized_metric(*args, **kwargs)
    metric.__name__ = "{metric}{args}{kwargs}".format(
        metric=parametrized_metric.__name__,
        args=format_args(args),
        kwargs=format_kwargs(kwargs)
    )
    return metric