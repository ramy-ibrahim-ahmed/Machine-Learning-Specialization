import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense
from keras.losses import CategoricalCrossentropy
from sklearn.datasets import make_blobs
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


### DATASET ###
np.random.seed(1270)
num_classes = 3
num_samples = 5
yhat = np.random.rand(num_samples, num_classes)
y = np.random.randint(num_classes, size=num_samples, dtype=int)
print(np.hstack([yhat, y.reshape(-1, 1)]))


### ONE-HOT ENCODING LABELS ###
y_hot = tf.one_hot(y, depth=num_classes)
print(y_hot)


### CATEGORICAL CROSS ENTROPY ###
J = CategoricalCrossentropy()
cost = J(y_hot, yhat)
print("Cost:", cost.numpy())


### NETWORK FOR ONE-HOT ENCODING LABELS ###
# LOAD DATA
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0, random_state=30)
# ENCODE LABELES
y_train_hot = tf.one_hot(y_train, depth=len(np.unique(y_train)))
model_hot = Sequential(
    [
        Dense(units=25, activation="relu"),
        Dense(units=15, activation="relu"),
        Dense(units=len(np.unique(y_train)), activation="linear"),
    ]
)
model_hot.compile(
    loss=CategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
)
model_hot.fit(X_train, y_train_hot, epochs=10)
# MAKE PREDICTIONS AS ONE-HOT ENCODED
yhat_hot = model_hot.predict(X_train)
for i in range(5):
    print(f"Example: {y_train_hot[i]}, Category: {np.argmax(yhat_hot[i])}")