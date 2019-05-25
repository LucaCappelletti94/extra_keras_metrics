extra_keras_metrics
==================================================================
Additional metrics integrated with the keras NN library, taken directly from `Tensorflow <https://www.tensorflow.org/api_docs/python/tf/metrics/>`_

How do I get this package?
----------------------------------------------
As usual, just install it with pip:

.. code:: bash

    pip install extra_keras_metrics


How do I use this package?
----------------------------------------------
Just by importing it you will be able to access all the non-parametric metrics, such as `auprc` and `auroc`:

.. code:: python

    import extra_keras_metrics

    model = my_keras_model()
    model.compile(
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=["auroc", "auprc"]
    )

For the parametric metrics, such as `average_precision_at_k`, you will need to import them, such as:

.. code:: python

    from extra_keras_metrics import average_precision_at_k

    model = my_keras_model()
    model.compile(
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=[average_precision_at_k(1), average_precision_at_k(2)]
    )

This way in the history of the model you will find both the metrics indexed as `average_precision_at_k_1` and `average_precision_at_k_2` respectively.

Which metrics do I get?
----------------------------------------------
You will get **all** the metrics from `Tensorflow <https://www.tensorflow.org/api_docs/python/tf/metrics/>`_