import tensorflow as tf
from tensorflow import keras
import numpy as np


class Network:

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
        self.model.fit(imgs, results, epochs=epochs)

    def predict(self, img_array):
        predictions = self.model.predict(img_array)
        return [np.argmax(predictions[i]) for i in range(5)]
