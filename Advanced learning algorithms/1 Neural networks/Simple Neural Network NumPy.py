import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import logging
from lab_coffee_utils import load_coffee_data

tf.autograph.set_verbosity(0)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
plt.style.use("./deeplearning.mplstyle")


### DATA DISCOVERY
X, y = load_coffee_data()
pos = y == 1
neg = y == 0
x1 = X[:, 0].reshape(-1, 1)
x2 = X[:, 1].reshape(-1, 1)
# plt.scatter(x1[pos], x2[pos], c="r", marker="x", label="GOOD")
# plt.scatter(x1[neg], x2[neg], c="b", marker="o", label="BAD")
# plt.title("Coffee Roasting", fontsize=12)
# plt.xlabel("Temperature", fontsize=12)
# plt.ylabel("Duration", fontsize=12)
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.legend(fontsize=12)
# plt.show()


### DATA NORMALIZATION ###
layer_norm = keras.layers.Normalization()
layer_norm.adapt(X)
Xn = layer_norm(X)


### SIGMOID ACTIVATION ###
def sigmoid(X, W, b):
    z = np.matmul(X, W) + b
    return 1.0 / (1.0 + np.exp(-z))


### DENSE LAYER ###
def dense(A_in, W, b, activation=sigmoid):
    units = W.shape[1]
    A_out = np.empty(units)
    for i in range(units):
        A_out[i] = activation(A_in, W[:, i], b[i])
    return A_out


### VICTORIZED DENSE LAYER ###
def DenseVictorized(A_in, W, b, activation=sigmoid):
    A_out = activation(A_in, W, b)
    return A_out


### SEQUENTIAL NETWORK ###
def sequential(X_in, *coef, layer=dense):
    A = X_in
    for i in range(0, len(coef), 2):
        W = coef[i]
        b = coef[i + 1]
        A = layer(A, W, b)
    return A


### NETWRK PREDICTIONS ###
def predict(X_in, *coef, network=sequential):
    m = X_in.shape[0]
    A_out = np.empty((m, 1))
    for i in range(m):
        A_out[i] = network(X_in[i], *coef)
    return A_out


### THRESHOLD ###
def threshold(P, threshold):
    return (P >= threshold).astype(int)


### USE LAYER ###
X_tst = np.array(
    [
        [200, 13.9],
        [200, 17],
    ]
)
X_tstn = layer_norm(X_tst)

W1 = np.array(
    [
        [-8.93, 0.29, 12.9],
        [-0.1, -7.32, 10.81],
    ]
)
b1 = np.array([-9.82, -9.28, 0.96])

W2 = np.array(
    [
        [-31.18],
        [-27.59],
        [-32.56],
    ]
)
b2 = np.array([15.41])

predictions = predict(X_tstn, W1, b1, W2, b2)
print(f"Percentages: {predictions}")
print(f"predictions: {threshold(predictions, 0.5)}")