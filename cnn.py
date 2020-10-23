# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:49:30 2018

@author: naufa
"""

from keras.datasets import mnist

mnist_data = mnist.load_data()

mnist_training = mnist_data[0]
mnist_testing = mnist_data[1]

# =============================================================================
# training_data = mnist_training[0]
# training_label = mnist_training[1]
# 
# testing_data = mnist_testing[0]
# testing_label = mnist_testing[1]
# =============================================================================

from sklearn.model_selection import train_test_split

training_data, testing_data, training_label, testing_label = train_test_split(mnist_training[0], mnist_training[1], test_size=0.1)

from matplotlib import pyplot as plt
index = 6000
plt.figure()
plt.title('label {}'.format(training_label[index]))
plt.imshow(training_data[index])


from keras.models import Sequential
from keras.activations import relu,softmax
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from keras.losses import categorical_crossentropy

training_data = training_data.reshape(training_data.shape[0], 28,28,1)
testing_data = testing_data.reshape(testing_data.shape[0], 28,28,1)
categorical_lab_train = np_utils.to_categorical(training_label)
categorical_lab_test = np_utils.to_categorical(testing_label)

model = Sequential()
model.add(Conv2D(filters= 8, kernel_size=(3,3), input_shape = (28, 28, 1), activation= relu))
model.add(Conv2D(filters= 8, kernel_size=(3,3), activation= relu))
model.add(Conv2D(filters= 8, kernel_size=(3,3), activation= relu))
model.add(MaxPool2D(pool_size=(2,2), strides=1))
model.add(Flatten())
model.add(Dense(units=50, activation=relu))
model.add(Dense(units=10, activation=softmax))
model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
model.fit(training_data, categorical_lab_train, epochs=1, batch_size=16)
acc_test = model.evaluate(testing_data, categorical_lab_test)
print('Akaurasi testing = {}'.format(acc_test))
print(model.summary())



