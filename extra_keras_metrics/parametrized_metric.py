from decorator import decorator
from typing import Callable, List, Dict, Tuple
import tensorflow as tf
import math


def format_int(param: int):
    return str(param)


def format_float(param: float):
    mantissa, integer = math.modf(param)
    return "{integer}_{mantissa}".format(
        integer=int(integer),
        mantissa=int(mantissa*10000)
    ).strip("0")


def format_parameter(param)->str:
    if isinstance(param, int):
        return format_int(param)
    if isinstance(param, float):
        return format_float(param)


def format_list(param: List):
    return "-".join(
        format_parameter(p) for p in param
    )


@decorator
def parametrized_metric(parametrized_metric: Callable[..., Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, Dict]]], *args: List)->Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, Dict]]:
    metric = parametrized_metric(*args)
    metric.__name__ = "{metric}{args}".format(
        metric=parametrized_metric.__name__,
        args=format_list(args)
    )
    return metric
