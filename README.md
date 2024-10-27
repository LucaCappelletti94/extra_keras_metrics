# Extra Keras Metrics

![PyPI version](https://badge.fury.io/py/extra-keras-metrics.svg)
[![python](https://img.shields.io/pypi/pyversions/extra-keras-metrics)](https://pypi.org/project/extra-keras-metrics/)
[![license](https://img.shields.io/pypi/l/extra-keras-metrics)](https://pypi.org/project/extra-keras-metrics/)
![Downloads](https://pepy.tech/badge/extra-keras-metrics)
[![mypy](https://github.com/LucaCappelletti94/extra_keras_metrics/actions/workflows/mypy.yml/badge.svg)](https://github.com/LucaCappelletti94/extra_keras_metrics/actions/)
[![Github Actions](https://github.com/LucaCappelletti94/extra_keras_metrics/actions/workflows/python.yml/badge.svg)](https://github.com/LucaCappelletti94/extra_keras_metrics/actions/)

Additional metrics integrated with the TensorFlow and Keras Neural Network libraries.

## How do I install this package?

As usual, just download it using pip:

```shell
pip install extra_keras_metrics
```

## How do I use this package?

In addition to importing individual metrics, sets of metrics are also available.

### Multi-class metrics

To retrieve an instance of the set of multi-class metrics, use:

```python
from extra_keras_metrics import get_minimal_multiclass_metrics

model = my_keras_model()
model.compile(
    optimizer="nadam",
    loss="categorical_crossentropy",
    metrics=get_minimal_multiclass_metrics()
)
```

### Sparse multi-class metrics

To retrieve an instance of the set of sparse multi-class metrics, use:

```python
from extra_keras_metrics import get_sparse_multiclass_metrics

model = my_keras_model()
model.compile(
    optimizer="nadam",
    loss="sparse_categorical_crossentropy",
    metrics=get_sparse_multiclass_metrics()
)
```

Note that currently, this only includes categorical accuracy, as it is the only one provided out-of-the-box by TensorFlow. More metrics are planned.

### Binary metrics

To retrieve an instance of the set of binary metrics, use:

```python
from extra_keras_metrics import get_standard_binary_metrics

model = my_keras_model()
model.compile(
    optimizer="nadam",
    loss="binary_crossentropy",
    metrics=get_standard_binary_metrics()
)
```

### All the binary metrics

We have implemented a wide range of binary metrics, including some lesser-known ones. To include **all** available binary metrics, use:

```python
from extra_keras_metrics import get_complete_binary_metrics

model = my_keras_model()
model.compile(
    optimizer="nadam",
    loss="binary_crossentropy",
    metrics=get_complete_binary_metrics()
)
```

## Extras

You might also enjoy these related packages:

- [extra_keras_utils](https://github.com/LucaCappelletti94/extra_keras_utils) - contains commonly used code for Keras projects.
- [plot_keras_history](https://github.com/LucaCappelletti94/plot_keras_history) - automatically plots Keras training history.
