# -*- coding: utf-8 -*-
"""
@Time    : 2017/10/12 13:37
@Author  : hubo
"""

from keras.models import Sequential
import keras
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
import os
import numpy as np


def load_testset():
    labels = ['oval', 'square', 'round', 'heart', 'diamond', 'pear', 'rectangle']
    '          0        1          2        3       4           5       6'
    imagedir = os.listdir('image-test')
    image_data_list = []
    label = []
    for img in imagedir:
        url = os.path.join('image-test', img)
        image = load_img(url, target_size=(128, 128))
        image_data_list.append(img_to_array(image))
        label.append(img.split('-')[0])
    img_data = np.array(image_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    return img_data, label

        
def load_dataset():
    """
    读取数据
    :param filedir:
    :return:
    """
    image_data_list = []
    label = []
    train_image_list = os.listdir('image-train')
    for img in train_image_list:
        url = os.path.join('image-train/' + img)
        image = load_img(url, target_size=(128, 128))
        image_data_list.append(img_to_array(image))
        label.append(img.split('-')[0])
    img_data = np.array(image_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    return img_data, label


def make_network():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(128, 128, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    return model


if __name__ == '__main__':
    train_x, train_y = load_dataset()
    train_y = np_utils.to_categorical(train_y)
    test_x, test_y = load_testset()
    test_y = np_utils.to_categorical(test_y) 
    model = make_network()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    hist = model.fit(train_x, train_y, batch_size=32, epochs=20, verbose=1)
    # save model
    model.save('face-classifier-with-test.f4')
    # print('Saved trained model')
    # Score trained model.
    # model = keras.models.load_model('face-classifier-2.f4')
    scores = model.evaluate(test_x, test_y, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
