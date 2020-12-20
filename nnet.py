import tensorflow as tf
from tensorflow import keras
import numpy as np


class Network:

    """
    Simple neural network for the identification of letters in images.
    """

    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(20, 20)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(26, activation=tf.nn.softmax)
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, imgs, results, epochs=5):
        """
        Train the network.
        :param imgs: np.array[(n, 20, 20)] = List of image to train
        :param results: np.array[n] = List of results to train
        :param epochs: int = Number of interations to train
        :return: void
        """
        self.model.fit(imgs, results, epochs=epochs)

    def predict(self, img_array):
        """
        Identify the character in an image.
        :param img_array: np.array[5, 20, 20] = List of images
        :return: List[int] = Number of identified chars
        """
        predictions = self.model.predict(img_array)
        return [np.argmax(predictions[i]) for i in range(5)]
