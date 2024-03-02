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


### RELU ###
def RELU(z):
    return np.maximum(0, z)


### SOFTMAX ###
def softmax(z):
    ex = np.exp(z)
    exs = np.sum(ex)
    return ex / exs


### DATASET ###
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(
    n_samples=2000, centers=centers, cluster_std=1.0, random_state=30
)


### NETWORK ###
model = Sequential(
    [
        Dense(25, activation="relu"),
        Dense(15, activation="relu"),
        Dense(4, activation="softmax"),
    ]
)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
)
model.fit(X_train, y_train, epochs=10)
# PROPABILITY
yhat = model.predict(X_train)
print(yhat[:2])
# RANGE BETWEEN 0, 1
print("largest value", np.max(yhat))
print("smallest value", np.min(yhat))


### PREFERED NETWORK ###
P_model = Sequential(
    [
        Dense(25, activation="relu"),
        Dense(15, activation="relu"),
        Dense(4, activation="linear"),
    ]
)
P_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
)
P_model.fit(X_train, y_train, epochs=10)
# PROPABILITY
P_yhat = P_model.predict(X_train)
print(P_yhat[:2])
# RANGE BETWEEN -n, n
print("largest value", np.max(P_yhat))
print("smallest value", np.min(P_yhat))
# APPLY SOFTMAX TO PREDICTIONS
y_softmax = tf.nn.softmax(P_yhat).numpy()
print(f"{y_softmax[:2]}")
print("largest value", np.max(y_softmax))
print("smallest value", np.min(y_softmax))
# GET CATEGORY
for i in range(5):
    print(f"{P_yhat[i]}, category: {np.argmax(P_yhat[i])}")