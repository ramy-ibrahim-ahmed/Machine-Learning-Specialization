import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


### DATASET ###
classes = 4
m = 100
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
std = 1.0
X, y = make_blobs(n_samples=m, centers=centers, cluster_std=std, random_state=30)


### DATA VISUALIZATION ###
# a = y == 0
# b = y == 1
# c = y == 2
# d = y == 3
# x1 = X[:, 0]
# x2 = X[:, 1]
# plt.scatter(x1[a], x2[a], c="b", label="Class 0")
# plt.scatter(x1[b], x2[b], c="r", label="Class 1")
# plt.scatter(x1[c], x2[c], c="y", label="Class 2")
# plt.scatter(x1[d], x2[d], c="g", label="Class 3")
# plt.legend(fontsize=12)
# plt.title("Multi-Class Data Representation", fontsize=16)
# plt.xlabel("Feature 1", fontsize=12)
# plt.ylabel("Feature 2", fontsize=12)
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.show()


### NETWORK ###
model = Sequential(
    [
        Dense(units=2, activation="relu"),
        Dense(units=4, activation="linear"),
    ]
)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
)
model.fit(X, y, epochs=200)