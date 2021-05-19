import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(Y_train.shape))
print('X_test shape: {}'.format(X_test.shape))
print('y_test shape: {}'.format(Y_train.shape))





