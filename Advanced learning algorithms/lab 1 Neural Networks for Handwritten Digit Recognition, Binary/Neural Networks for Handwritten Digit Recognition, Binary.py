import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense
from autils import *
from public_tests import *
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


### DATA LOADING ###
X, y = load_data()


### DATA VISUALIZATION ###
# m = X.shape[0]
# fig, axes = plt.subplots(8, 8)
# fig.tight_layout(pad=0.1)
# for i, ax in enumerate(axes.flat):
#     random_index = np.random.randint(m)
#     x_random = X[random_index]
#     x_random_reshaped = x_random.reshape(20, 20).T
#     ax.imshow(x_random_reshaped, cmap="gray")
#     ax.set_title(y[random_index, 0])
#     ax.set_axis_off()
# plt.show()


### NETWORK BUILDING ###
model = Sequential(
    [
        keras.Input(shape=(400,)),
        Dense(units=25, activation="sigmoid", name="L1"),
        Dense(units=15, activation="sigmoid", name="L2"),
        Dense(units=1, activation="sigmoid", name="L3"),
    ],
    name="M1",
)
model.summary()
test_c1(model)  # --> UNIT TESTS #


### NETWORK TRAINING ###
model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
)
model.fit(X, y, epochs=20)


### MAKE PREDICTIONS ###
y_hat = model.predict(X[500].reshape(1, -1))
print(f"Percentage: {y_hat[0, 0]}")
print(f"Class: {(y_hat[0, 0] >= 0.5).astype(int)}")


### VISUALIZATION ###
# fig, axes = plt.subplots(8, 8, figsize=(8, 8))
# fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92])
# for i, ax in enumerate(axes.flat):
#     index = np.random.randint(X.shape[0])
#     x_random = X[index].reshape(20, 20).T
#     ax.imshow(x_random, cmap="gray")
#     yhat = ((model.predict(x_random.reshape(1, -1))) >= 0.5).astype(int)
#     ax.set_title(f"{y[index, 0]}, {yhat[0, 0]}")
#     ax.set_axis_off()
# fig.suptitle("Y, Y^", fontsize=16)
# plt.show()


### SEE ERRORS ###
predictions = model.predict(X)
predictions = (predictions >= 0.5).astype(int)
errors = np.where(y != predictions)

random_index = errors[0][0]
X_random_reshaped = X[random_index].reshape((20, 20)).T

# plt.figure()
# plt.imshow(X_random_reshaped, cmap="gray")
# plt.title(f"{y[random_index,0]}, {predictions[random_index, 0]}")
# plt.show()