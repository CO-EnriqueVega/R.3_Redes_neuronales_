#!/usr/bin/env python
# coding: utf-8

# Digit recognition


from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras.utils import to_categorical

batch_size = 128
num_classes = 10
epochs = 5

# Image size (pixels)
img_rows, img_cols = 28, 28

# Data set split
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Pre-processing
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Label pre-processing
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print('Size of x_train: ', x_train.shape)
print(x_train.shape[0], 'examples for training')
print(x_test.shape[0], 'examples for testing')

#---------- DESIGN (PLEASE, ONLY MODIFY THIS CELL) --------------------
model = Sequential()
model.add(Conv2D(128, kernel_size=(6, 6), activation='relu', input_shape=input_shape))   # Modified filters and Kernel size
model.add(Conv2D(256, (3, 3), activation='relu'))    # Modified filters
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.40))    # Increased dropout rate to 40%
model.add(Flatten())
model.add(Dense(512, activation='relu'))     # Increased number of neurons
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
#--------------- END OF DESIGN ---------------------------------------

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=['accuracy'])
              
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
          
score = model.evaluate(x_test, y_test, verbose=0)

print('Error in test set:', '{:3.3f}'.format(score[0]))
print('Accuracy using test set:', '{:3.3f}'.format(score[1]))