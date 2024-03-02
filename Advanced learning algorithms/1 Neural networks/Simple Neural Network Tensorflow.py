import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import logging
from keras.models import Sequential
from keras.layers import Dense
from lab_coffee_utils import load_coffee_data

tf.autograph.set_verbosity(0)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


### DATA SET ###
X, y = load_coffee_data()


### PLOT DATA ###
# pos = y == 1
# neg = y == 0
# x1 = X[:, 0].reshape(-1, 1)
# x2 = X[:, 1].reshape(-1, 1)
# plt.scatter(x1[pos], x2[pos], c="r", marker="x", label="GOOD")
# plt.scatter(x1[neg], x2[neg], c="b", marker="o", label="BAD")
# plt.ylabel("Duration", fontsize=12)
# plt.xlabel("Temperature", fontsize=12)
# plt.title("Coffe Roasting", fontsize=12)
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.legend(fontsize=12)
# plt.show()


### DATA NORMALIZATION ###
# print(np.ptp(X, axis=0))
layer_norm = keras.layers.Normalization(axis=-1)
layer_norm.adapt(X)
Xn = layer_norm(X)
# print(np.ptp(Xn, axis=0))


### DATA TILE ###
# print(X.shape, y.shape)
Xt = np.tile(Xn, (1000, 1))
yt = np.tile(y, (1000, 1))
# print(Xt.shape, yt.shape)


### BUILD NETWORK ###
tf.random.set_seed(1234)
model = Sequential(
    [
        keras.Input(shape=(2,)),
        Dense(3, activation="sigmoid", name="L1"),
        Dense(1, activation="sigmoid", name="L2"),
    ]
)


### NETWORK SUMMARY ###
# model.summary()


### GET LAYERS WEIGHTS ###
W1, b1 = model.get_layer("L1").get_weights()
W2, b2 = model.get_layer("L2").get_weights()
# print(f"W1{W1.shape}", f"\nb1{b1.shape}:")
# print(f"W2{W2.shape}", f"\nb2{b2.shape}:")


### NETWORK COMPILE ###
model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
)


### TRAIN NETWORK ###
model.fit(Xt, yt, epochs=10)


### SET PREVIOUS COMPUTED COEFFICIENTS ###
W1 = np.array([[-8.94, 0.29, 12.89], [-0.17, -7.34, 10.79]])
b1 = np.array([-9.87, -9.28, 1.01])
W2 = np.array([[-31.38], [-27.86], [-32.79]])
b2 = np.array([15.54])
model.get_layer("L1").set_weights([W1, b1])
model.get_layer("L2").set_weights([W2, b2])


### MAKE PREDICTION ###
X_test = np.array(
    [
        [200, 13.9],
        [200, 17],
    ]
)
X_test_norm = layer_norm(X_test)
percentage = model.predict(X_test_norm)
# print(f"Percentage: {percentage}")
prediction = (percentage >= 0.5).astype(int)
# print(f"calss: {prediction}")