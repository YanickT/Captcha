import random
import os
import numpy as np
from PIL import Image

# own scripts
import nnet
import prepare as pre


TRAINPATH = "train_data"


chr_dict = {i: chr(ord('a') + i) for i in range(26)}
ord_dict = {chr(ord('a') + i): i for i in range(26)}


def main():
    """
    Train and test network.
    Uses the data in the train_data folder.
    """
    # load data
    data = os.listdir(TRAINPATH)
    random.shuffle(data)

    # split into test and trainings data
    train_data, test_data = data[:-50], data[-50:]

    # setup trainings data
    imgs_array = np.zeros((len(train_data) * 5, 20, 20), np.int32)
    results_array = np.zeros(len(train_data) * 5, np.int32)
    for data_index, data in enumerate(train_data):
        img = Image.open(TRAINPATH + "/" + data)
        chars = pre.extract_chars(img)
        solution = data.split("'")[1]

        for char_index, char in enumerate(chars):
            imgs_array[5 * data_index + char_index, :, :] = char
            results_array[5 * data_index + char_index] = ord_dict[solution[char_index]]

    # setup network and train it
    net = nnet.Network()
    net.train(imgs_array, results_array, 8)

    # test network
    test_result = {True: 0, False: 0}
    img_array = np.zeros((5, 20, 20), np.int32)
    for data_index, data in enumerate(test_data):
        img = Image.open(TRAINPATH + "/" + data)
        chars = pre.extract_chars(img)
        solution = data.split("'")[1]

        for char_index, char in enumerate(chars):
            img_array[char_index, :, :] = char

        results = net.predict(img_array)
        results = "".join([chr_dict[result] for result in results])
        test_result[solution == results] += 1

    return test_result[True]


if __name__ == "__main__":
    values = [main() for i in range(30)]
    print(values)
