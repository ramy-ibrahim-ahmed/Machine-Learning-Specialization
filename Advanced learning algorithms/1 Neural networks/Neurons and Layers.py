import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from keras import Sequential
from keras.layers import Dense
from keras.activations import sigmoid

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


### INITIALIZE DATA ###
X_train = np.array(
    [
        [1.0],
        [2.0],
    ],
    dtype=np.float32,
)
Y_train = np.array(
    [
        [300.0],
        [500.0],
    ],
    dtype=np.float32,
)


### VISUALIZE DATA ###
# plt.scatter(X_train, Y_train, marker="x", c="r", label="Data Points")
# plt.ylabel("Price (in 1000s of dollars)", fontsize="xx-large")
# plt.xlabel("Size (1000 sqft)", fontsize="xx-large")
# plt.title("Data Points", fontsize="xx-large")
# plt.legend(fontsize=12)
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.show()


### INITIALIZE A LAYER ###
linear_layer = Dense(units=1, activation="linear")


### GET LAYER WEIGHTS ###
# print(linear_layer.get_weights())  # empty


### FEED LAYER ###
a1 = linear_layer(X_train[0].reshape(-1, 1))
w, b = linear_layer.get_weights()
# print(a1)
# print(f"w = {w}, b={b}")


### SET WEIGHTS ###
new_w = np.array(
    [
        [200],
    ]
)
new_b = np.array([100])
linear_layer.set_weights([new_w, new_b])
# print(linear_layer.get_weights())


### MAKE PREDICTION ###
a1 = linear_layer(X_train[0].reshape(-1, 1))
a_scratch = np.dot(new_w, X_train[0].reshape(-1, 1)) + new_b
# print(f"layer: {a1[0, 0]}")
# print(f"equation: {a_scratch[0, 0]}")


### PLOT ALL PREDICTIONS ###
# y_pred = linear_layer(X_train)
# plt.plot(X_train, y_pred, label="Linear function")
# plt.scatter(X_train, y_pred, c="r", marker="x", label="Data points")
# plt.xlabel("Price (in 1000s of dollars)", fontsize="xx-large")
# plt.ylabel("Size (1000 sqft)", fontsize="xx-large")
# plt.title("Linear Function and Data Points", fontsize="xx-large")
# plt.legend(fontsize=12)
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.show()


### USE LOGISTIC REGRESSION NEURON ###
### TRAIN DATA & CONDITIONS ###
X_train = np.array([0.0, 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1, 1)
Y_train = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32).reshape(-1, 1)
pos = Y_train == 1
neg = Y_train == 0


### PLOT DATA ###
# plt.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c="r", label="Positive")
# plt.scatter(X_train[neg], Y_train[neg], marker='o', s=100, c="b", label="Negative", facecolors="none", edgecolors=dlc["dlblue"], lw=3)
# plt.xlabel("X", fontsize=12)
# plt.ylabel("Y", fontsize=12)
# plt.title("Two Classes + & -", fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.show()


### INITIALIZE LOGISTIC NEURONS ###
model = Sequential(
    [Dense(1, input_dim=1, activation=sigmoid, name="L1")]
)


### NETWORK SUMMARY ###
# model.summary()


### GET LAYER FROM A NETWORK ###
logistic_layer = model.get_layer("L1")
w, b = logistic_layer.get_weights()
set_w = np.array([[2]])
set_b = np.array([-4.5])
model.set_weights([set_w, set_b])


### PREDICT BY THE NETWORK ###
a1 = model.predict(X_train[0].reshape(-1, 1))
# print(f"{a1[0, 0]:.2f}")


### PLOT NETWORK ###
# plt.plot(X_train, model.predict(X_train), label='Sigmoid Function')
# plt.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c="r", label="Positive")
# plt.scatter(X_train[neg], Y_train[neg], marker='o', s=100, c="b", label="Negative", facecolors="none", edgecolors=dlc["dlblue"], lw=3)
# plt.xlabel('x')
# plt.ylabel('sigmoid(x)')
# plt.title('Sigmoid Function and Decision Boundary')
# plt.legend()
# plt.grid(True)
# plt.show()