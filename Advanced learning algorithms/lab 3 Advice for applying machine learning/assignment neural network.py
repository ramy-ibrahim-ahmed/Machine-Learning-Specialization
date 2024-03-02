import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.regularizers import l2
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from public_tests_a1 import *
from assigment_utils import *


tf.keras.backend.set_floatx("float64")
tf.autograph.set_verbosity(0)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


### Dataset ###
X, y, centers, classes, std = gen_blobs()
xtrain, X_, ytrain, y_ = train_test_split(X, y, test_size=0.50, random_state=1)
xdev, xtest, ydev, ytest = train_test_split(X_, y_, test_size=0.20, random_state=1)
show = False
if show:
    classes = np.unique(y)
    colors = plt.get_cmap("tab10").colors
    for i, c in enumerate(classes):
        indexs = np.where(y == c)
        plt.scatter(X[indexs, 0], X[indexs, 1], marker="o", c=colors[i], label=f"c{c}")
    plt.title("Classes", fontsize=16)
    plt.legend()
    plt.grid(True, linestyle="--", lw=0.5)
    plt.tight_layout()
    plt.show()


### CLASSIFICATION ERROR ###
def categorical_error(y, yhat):
    return np.mean(y != yhat)


### TEST ###
test = False
if test:
    y_hat = np.array([1, 2, 0])
    y_tmp = np.array([1, 2, 3])
    y_hat = np.array([[1], [2], [0], [3]])
    y_tmp = np.array([[1], [2], [1], [3]])
    test_eval_cat_err(categorical_error)


### COMPLEX NETWORK ###
tf.random.set_seed(1234)
network_c = Sequential(
    [
        Dense(units=120, activation=relu),
        Dense(units=40, activation=relu),
        Dense(units=6, activation=linear),
    ],
    name="complix",
)
network_c.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(0.01),
)
go = False
if go:
    network_c.fit(xtrain, ytrain, epochs=1000)
    model_test(network_c, classes, xtrain.shape[1])
    predict = lambda X: np.argmax(tf.nn.softmax(network_c.predict(X)).numpy(), axis=1)
    Jtrain = categorical_error(ytrain, predict(xtrain))
    Jdev = categorical_error(ydev, predict(xdev))
    print(f"Training error in complex network: {Jtrain:0.3f}")
    print(f"Development error in complex network: {Jdev:0.3f}")


### SIMPLE MODEL ###
network_s = Sequential(
    [
        Dense(units=6, activation=relu),
        Dense(units=6, activation=linear),
    ],
    name="simple",
)
network_s.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(0.01),
)
go = False
if go:
    network_s.fit(xtrain, ytrain, epochs=1000)
    predict = lambda X: np.argmax(tf.nn.softmax(network_s.predict(X)).numpy(), axis=1)
    Jtrain = categorical_error(ytrain, predict(xtrain))
    Jdev = categorical_error(ydev, predict(xdev))
    print(f"Training error in simple network: {Jtrain:0.3f}")
    print(f"Development error in simple network: {Jdev:0.3f}")


### COMPLEX NETWORK WITH REGULARIZATION ###
network_r = Sequential(
    [
        Dense(units=120, activation=relu, kernel_regularizer=l2(0.1)),
        Dense(units=40, activation=relu, kernel_regularizer=l2(0.1)),
        Dense(units=6, activation=linear),
    ],
    name="regularized",
)
network_r.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(0.01),
)
go = False
if go:
    network_r.fit(xtrain, ytrain, epochs=1000)
    model_r_test(network_r, classes, xtrain.shape[1])
    predict = lambda X: np.argmax(tf.nn.softmax(network_s.predict(X)).numpy(), axis=1)
    Jtrain = categorical_error(ytrain, predict(xtrain))
    Jdev = categorical_error(ydev, predict(xdev))
    print(f"Training error in simple network: {Jtrain:0.3f}")
    print(f"Development error in simple network: {Jdev:0.3f}")


### TRY MANY DIFFERENT REGULARIZATION LAMBDAS ###
lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
Jtrains = []
Jdevs = []
go = False
if go:
    for i in range(len(lambdas)):
        lambda_ = lambdas[i]
        model = Sequential(
            [
                Dense(
                    120,
                    activation="relu",
                    kernel_regularizer=l2(lambda_),
                ),
                Dense(
                    40,
                    activation="relu",
                    kernel_regularizer=l2(lambda_),
                ),
                Dense(classes, activation="linear"),
            ]
        )
        model.compile(
            loss=SparseCategoricalCrossentropy(from_logits=True),
            optimizer=Adam(0.01),
        )
        model.fit(xtrain, ytrain, epochs=1000)
        predict = lambda X: np.argmax(tf.nn.softmax(model.predict(X)).numpy(), axis=1)
        Jtrain = categorical_error(ytrain, predict(xtrain))
        Jdev = categorical_error(ydev, predict(xdev))
        Jtrains.append(Jtrain)
        Jdevs.append(Jdev)
        print(f"Finished lambda = {lambda_}")
    plt.plot(lambdas, Jtrains, c="r", marker=".", lw=2, label="Train MSE")
    plt.plot(lambdas, Jdevs, marker=".", lw=2, label="Development MSE")
    plt.axvline(lambdas[np.argmin(Jdevs)], linestyle="--", lw=1, c="black")
    plt.title("Tunning Regularization", fontsize=16)
    plt.xlabel("Lambda", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", lw=0.5)
    plt.tight_layout()
    plt.show()