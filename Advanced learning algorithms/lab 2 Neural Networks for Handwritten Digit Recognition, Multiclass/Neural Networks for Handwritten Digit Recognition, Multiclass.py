import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import logging
from keras import Input, Sequential
from keras.layers import Dense
from keras.activations import linear, relu, softmax
from public_tests import *
from autils import *

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


### DATASET ###
X, y = load_data()


### DATA VISUALIZATION ###
# m = X.shape[0]
# fig, axis = plt.subplots(8, 8)
# fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]
# widgvis(fig)
# for i, ax in enumerate(axis.flat):
#     random_index = np.random.randint(m)
#     x_random = X[random_index].reshape(20, 20).T
#     ax.imshow(x_random, cmap="gray")
#     ax.set_title(y[random_index, 0])
#     ax.set_axis_off()
# fig.suptitle("Dataset True Labels", fontsize=12)
# plt.show()


### NETWORK ###
tf.random.set_seed(1234)
# NETWORK ARCHTECTURE
model = Sequential(
    [
        Input(shape=(400,)),
        Dense(units=25, activation=relu, name="L1"),
        Dense(units=15, activation=relu, name="L2"),
        Dense(units=10, activation=linear, name="L3"),
    ],
    name="M1",
)
# MODEL SUMMARY
# model.summary()
# UNIT TEST
test_model(model, 10, 400)
# NETWORK TRAINING
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
)
J = model.fit(X, y, epochs=40)


### COST VISUALIZATION ###
# plt.plot(J.history['loss'], label='loss')
# plt.ylim([0, 2])
# plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('loss (cost)', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.title("Sparse Categorical Cross Entropy", fontsize=16)
# plt.tight_layout()
# plt.show()


### PREDICTIONS ###
# index = np.where(y == 7)
# index = index[0][0]
# index = np.random.randint(5000)
# yhat = model.predict(X[index].reshape(1, -1))
# yhat_val = np.argmax(yhat)
# image = X[index].reshape(20, 20).T
# plt.imshow(image, cmap="gray")
# plt.title(f"True Label: {y[index, 0]}, Predicted Label: {yhat_val}", fontsize=14)
# plt.show()


### VISUALIZE PREDICTIONS ###
# fig, axes = plt.subplots(8, 8)
# fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]
# widgvis(fig)
# for i, ax in enumerate(axes.flat):
#     random_index = np.random.randint(X.shape[0])
#     x_random = X[random_index].reshape(20, 20).T
#     yhat_random = model.predict(X[random_index].reshape(1, -1))
#     yhat_random = tf.nn.softmax(yhat_random)
#     yhat_random = np.argmax(yhat_random)
#     ax.imshow(x_random, cmap="gray")
#     ax.set_title(f"{y[random_index, 0]}, {yhat_random}")
#     ax.set_axis_off()
# plt.show()