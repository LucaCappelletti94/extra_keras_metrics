from keras import Sequential
from keras.layers import Dense
from typing import Callable, Tuple
import tensorflow as tf
import numpy as np

def train_with_metric(metric:Callable[[tf.Tensor, tf.Tensor], float], size:Tuple[int, int]=(10,1)):
    train = np.random.uniform(size=size), np.random.randint(2, size=size)
    test = np.random.uniform(size=size), np.random.randint(2, size=size)
    model = Sequential([Dense(units=1, activation="sigmoid")])
    model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=[metric])
    model.fit(*train, verbose=0, batch_size=size[0], validation_data=test)