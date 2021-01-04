import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from extra_keras_metrics import get_binary_metrics


def test_model_run():
    model = Sequential([
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=get_binary_metrics()
    )
    model.fit(
        np.random.uniform(size=(10, 10)),
        np.random.randint(2, size=(10, )),
    )
