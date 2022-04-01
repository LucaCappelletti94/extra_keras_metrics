import numpy as np
from extra_keras_metrics import (get_complete_binary_metrics,
                                 get_minimal_multiclass_metrics)
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def test_binary_model_run():
    """Test that all metrics actually work when used with a Keras model."""
    model = Sequential([
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=get_complete_binary_metrics()
    )
    history = model.fit(
        np.random.uniform(size=(1000, 10)),
        np.random.randint(2, size=(1000, )),
        epochs=30,
        sample_weight=np.random.uniform(size=(1000, )),
    )


def test_multiclass_model_run():
    """Test that all metrics actually work when used with a Keras model."""
    model = Sequential([
        Dense(5, activation="softmax")
    ])
    model.compile(
        optimizer="nadam",
        loss="categorical_crossentropy",
        metrics=get_minimal_multiclass_metrics()
    )
    model.fit(
        np.random.uniform(size=(10, 10)),
        np.random.randint(2, size=(10, 5)),
    )
