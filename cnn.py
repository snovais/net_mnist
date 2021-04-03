# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 18:43:37 2021

@author: sergi

https://www.tensorflow.org/tutorials/quickstart/advanced
"""

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Model

import processing as pr


# load data and prepare
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, y_train = pr.canny_filter(x_train, y_train)
x_test, y_test = pr.canny_filter(x_test, y_test)
x_train, x_test = x_train / 255.0, x_test / 255.0


# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices(
                                  (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)



class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 5, activation='relu')
    self.pool1 = MaxPool2D(pool_size = (16, 16))
    self.conv2 = Conv2D(32, 5, activation='relu')
    self.pool1 = MaxPool2D(pool_size = (16, 16))
    self.flatten = Flatten()
    self.d1 = Dense(12800, activation='relu')
    self.dr1 = Dropout(0.3)
    self.d2 = Dense(1024, activation='relu')
    self.dr2 = Dropout(0.3)
    self.d3 = Dense(1024, activation='relu')
    self.dr3 = Dropout(0.3)
    self.d4 = Dense(1024, activation='relu')
    self.dr4 = Dropout(0.3)
    self.d5 = Dense(1024, activation='relu')
    self.dr5 = Dropout(0.3)
    self.d6 = Dense(10)


  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.dr1(x)
    x = self.d2(x)
    x = self.dr2(x)
    x = self.d3(x)
    x = self.dr3(x)
    x = self.d4(x)
    x = self.dr4(x)
    x = self.d5(x)

    return self.d6(x)


# Create an instance of the model
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adamax()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
  

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)
  
  
EPOCHS = 40

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )


  