# -*- coding: utf-8 -*-

"""
Reconhecimento de digitos manuscritos utilizando Redes neurais Artificiais Densas
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

import cv2

import matplotlib.pyplot as plt

import time


# instanciando o otimizador
optimizer = keras.optimizers.Adamax(learning_rate = 1e-3)


# instanciando a função de perda
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# preparando o banco de imagens mnist
batch_size = 128
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

def canny(x, y):
    edges = []
    for i in range(len(x)):
        edges.append(cv2.Canny(x[i], 28, 28))
   
    del x
        
    return np.array(edges), y
        
x_train, y_train = canny(x_train, y_train)
x_test, y_test = canny(x_test, y_test)

print(y_train[0])


"""
# imagens vem com formato (x, 28, 28, 1), o código abaixo 
# transforma para matrizes 2D
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))


# reserva 10.000 amostras para validação
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]


# Prepara o banco de imagens para formato tensorflow
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# separa um buffer para sempre ter dados disponíveis para a rede
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# preparar os dados para validação
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)

# construção do modelo
# função RELU transforma todos os dados para valores acima de 0


def build_model( units, drop_out ):
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(units, activation="relu", name="dense_1")(inputs)
    x = layers.Dropout(drop_out)(inputs)
    x = layers.Dense(units, activation="relu", name="dense_2")(x)
    x = layers.Dropout(drop_out)(x)
    x = layers.Dense(units, activation="relu", name="dense_3")(x)
    x = layers.Dropout(drop_out)(x)
    x = layers.Dense(units, activation="relu", name="dense_4")(x)
    x = layers.Dropout(drop_out)(x)
    x = layers.Dense(units, activation="relu", name="dense_5")(x)
    x = layers.Dropout(drop_out)(x)
    x = layers.Dense(units, activation="relu", name="dense_6")(x)
    x = layers.Dropout(drop_out)(x)
    x = layers.Dense(units, activation="relu", name="dense_7")(x)
    x = layers.Dropout(drop_out)(x)
    outputs = layers.Dense(10, name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs)

model = build_model(1024, 0.3)

# preparando as métricas
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()


@tf.function
def train_step( x, y ):
    # x -> 64, 784
    # y -> 64

    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value


@tf.function
def test_step( x, y ):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)
    

def learning( epochs ):
    for epoch in range(epochs):
        print("\nInício da época %d" % (epoch,))
        start_time = time.time()

        # itera sobre os lotes do dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train)

            # mostra log a cada 200 passos
            if step % 200 == 0:
                print(
                    "Perda de treinamento (para um pacote) no passo %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Visto até agora: %d amostras" % ((step + 1) * 128))

        # mostra métricas no fim de cada época
        train_acc = train_acc_metric.result()
        print("Acurácia do treinamento após época: %.4f" % (float(train_acc),))

        # reseta estado no fim de cada época
        train_acc_metric.reset_states()

        # roda loop para testar os dados de validação no fim de cada época
        for x_batch_val, y_batch_val in val_dataset:
            test_step(x_batch_val, y_batch_val)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Acurácia na Validação: %.4f" % (float(val_acc),))
        print("tempo decorrido: %.2fs" % (time.time() - start_time))


learning( epochs = 1500 )
"""