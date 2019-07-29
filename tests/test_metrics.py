from extra_keras_metrics import metrics, parametric_metrics, non_parametric_metrics
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import cifar10
from extra_keras_utils import set_seed

params = {
    "average_precision_at_k": {"args": [1, ], "kwargs": {}},
    "precision_at_k": {"args": [], "kwargs": {"k": 1}},
    "precision_at_thresholds": {"args": [[0.5], ], "kwargs": {}},
    "recall_at_k": {"args": [1, ], "kwargs": {}},
    "recall_at_thresholds": {"args": [[0.5], ], "kwargs": {}},
    "sensitivity_at_specificity": {"args": [0.5, ], "kwargs": {}},
    "specificity_at_sensitivity": {"args": [0.5, ], "kwargs": {}},
    "false_negatives_at_thresholds": {"args": [[0.5], ], "kwargs": {}},
    "true_negatives_at_thresholds": {"args": [[0.5], ], "kwargs": {}},
    "false_positives_at_thresholds": {"args": [[0.5], ], "kwargs": {}},
    "true_positives_at_thresholds": {"args": [[0.5], ], "kwargs": {}},
    "mean_cosine_distance": {"args": [1, ], "kwargs": {}},
    "mean_iou": {"args": [2, ], "kwargs": {}},
    "mean_per_class_accuracy": {"args": [5, ], "kwargs": {}},
    "mean_relative_error": {"args": [tf.constant(np.ones([1000, 10]), dtype=np.float32), ], "kwargs": {}},
}


def test_metrics():
    global params
    set_seed(60, True)
    assert all([
        m.__name__ in params for m in parametric_metrics
    ]) and all([
        m.__name__ not in params for m in non_parametric_metrics
    ])
    ready_metrics = [
        metric(
            *params[metric.__name__]["args"],
            **params[metric.__name__]["kwargs"]
        ) if metric.__name__ in params else metric.__name__ for metric in metrics
    ]
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train, y_train = x_train[:10000], y_train[:10000]

    # Convert class vectors to binary class matrices.
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", padding='same', input_shape=x_train.shape[1:]),
        Flatten(),
        Dense(100, activation="relu"),
        Dense(100, activation="relu"),
        Dense(num_classes, activation="sigmoid")
    ])
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=ready_metrics
    )
    history = pd.DataFrame(
        model.fit(
            x_train,
            y_train,
            verbose=0,
            epochs=20,
            batch_size=1000,
            shuffle=True,
            validation_data=(x_test, y_test)
        ).history
    )
    assert np.all(history.var() != 0)