from keras import backend as K
import tensorflow as tf
from typing import Callable, Tuple, Dict
from decorator import decorator

@decorator
def metric(metric:Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, Dict]], labels:tf.Tensor, predictions:tf.Tensor)->Dict:
	"""Wrap given metric for being used in Keras.
		metric:Callable[[tf.Tensor, tf.Tensor], float], metric to be wrapped.
		labels:tf.Tensor, the expected output values.
		predictions:tf.Tensor, the predicted output values.
	"""
	_, update_op = metric(labels, predictions)
	K.get_session().run(tf.local_variables_initializer())
	return update_op