from keras import Sequential
from keras.layers import Dense
import extra_keras_metrics
import numpy as np


def test_keras_injection():
    model = Sequential([
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=["auprc", "acc", "mae"],
    )

    model.fit(
        *np.random.randint(2, size=(2, 1000, 1)),
        verbose=0
    )