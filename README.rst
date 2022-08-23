Extra Keras Metrics
=========================================================================================
|pip| |downloads|

Additional metrics integrated with the TensorFlow and Keras Neural Network libraries.

How do I install this package?
----------------------------------------------
As usual, just download it using pip:

.. code:: shell

    pip install extra_keras_metrics


How do I use this package?
----------------------------------------------
Other than by importing the single metrics from the package, we make available
also sets of metrics.

Multi-class metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To retrieve an instance of the set of multi-class metrics you can use:

.. code:: python

    from extra_keras_metrics import get_minimal_multiclass_metrics

    model = my_keras_model()
    model.compile(
        optimizer="nadam",
        loss="categorical_crossentropy",
        metrics=get_minimal_multiclass_metrics()
    )

Sparse multi-class metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To retrieve an instance of the set of sparse multi-class metrics you can use:

.. code:: python

    from extra_keras_metrics import get_sparse_multiclass_metrics

    model = my_keras_model()
    model.compile(
        optimizer="nadam",
        loss="sparse_categorical_crossentropy",
        metrics=get_sparse_multiclass_metrics()
    )

Note that for now this only includes the categorial accuracy, since it is the only one
provided out-of-the-box by Tensorflow. We will be implementing more metrics ourselves.

Binary metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To retrieve an instance of the set of binary-class metrics you can use:

.. code:: python

    from extra_keras_metrics import get_standard_binary_metrics

    model = my_keras_model()
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=get_standard_binary_metrics()
    )

All the binary metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We have implemented all sorts of binary metrics, including some relatively
more obscure ones. If you want ALL the binary metrics we implemented you can
use the following method:

.. code:: python

    from extra_keras_metrics import get_complete_binary_metrics

    model = my_keras_model()
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=get_complete_binary_metrics()
    )

Extras
----------------------------
I've created also another couple packages you might enjoy this other one,
called `extra_keras_utils <https://github.com/LucaCappelletti94/extra_keras_utils>`_
that contains some commonly used code for Keras projects and
`plot_keras_history <https://github.com/LucaCappelletti94/plot_keras_history>`_
which automatically plots a Keras training history.


.. |pip| image:: https://badge.fury.io/py/extra-keras-metrics.svg
    :target: https://badge.fury.io/py/extra_keras_metrics
    :alt: Pypi project

.. |downloads| image:: https://pepy.tech/badge/extra-keras-metrics
    :target: https://pepy.tech/badge/extra-keras-metrics
    :alt: Pypi total project downloads 