import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import Dense
from keras.activations import relu, linear
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


os.environ["CUDA_VISIBLE_DEVICES"] = (r"PCI\VEN_10DE&DEV_1F95&SUBSYS_3FA317AA&REV_A1\4&11F4856E&0&0008")
np.set_printoptions(precision=2)
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)
tf.random.set_seed(1234)
np.random.seed(1234)


### DATASET ###
data = np.loadtxt(r"./data/data_w3_ex2.csv", delimiter=",")
x = data[:, :-1]
y = data[:, -1]
y = np.expand_dims(y, axis=1)


### PLOT DATA ###
# pos = y == 1
# neg = y == 0
# x1 = x[:, 0].reshape(-1, 1)
# x2 = x[:, 1].reshape(-1, 1)
# plt.scatter(x1[pos], x2[pos], c="r", marker="x", label="y=0")
# plt.scatter(x1[neg], x2[neg], label="y=1")
# plt.xlabel("X1", fontsize=12)
# plt.ylabel("X2", fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.show()


### DATA SPLIT ###
xtrain, xtemp, ytrain, ytemp = train_test_split(x, y, test_size=0.40, random_state=1234)
xdev, xtest, ydev, ytest = train_test_split(xtemp, ytemp, test_size=0.50, random_state=1234)
del xtemp, ytemp


### FEATURE SCALE ###
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xdev = scaler.transform(xdev)
xtest = scaler.transform(xtest)


### CLASSIFICATION ERROR BY FRACTION COST ###
probability = np.array([0.2, 0.3, 0.7, 0.3, 0.8])
prediction = np.where(probability >= 0.5, 1, 0)
truth = np.array([1, 0, 1, 1, 1])
J = np.mean(prediction != truth)
print(J)


### BUILD NETWORKS ###
network1 = Sequential(
    [
        Input(shape=(2,)),
        Dense(units=25, activation=relu),
        Dense(units=15, activation=relu),
        Dense(units=1, activation=linear),
    ],
    name="network1",
)
network2 = Sequential(
    [
        Input(shape=(2,)),
        Dense(units=20, activation=relu),
        Dense(units=12, activation=relu),
        Dense(units=12, activation=relu),
        Dense(units=20, activation=relu),
        Dense(units=1, activation=linear),
    ],
    name="network2",
)
network3 = Sequential(
    [
        Input(shape=(2,)),
        Dense(units=32, activation=relu),
        Dense(units=16, activation=relu),
        Dense(units=8, activation=relu),
        Dense(units=4, activation=relu),
        Dense(units=12, activation=relu),
        Dense(units=1, activation=linear),
    ],
    name="network3",
)
nets = [network1, network2, network3]


### TRAIN & EVALUATE & SELECT NETWORKS ###
Jtrains = []
Jdevs = []
for net in nets:
    net.compile(
        loss=BinaryCrossentropy(from_logits=True),
        optimizer=Adam(learning_rate=0.01),
    )
    net.fit(xtrain, ytrain, epochs=200, verbose=0)
    yhat = net.predict(xtrain)
    yhat = tf.math.sigmoid(yhat)
    yhat = np.where(yhat >= 0.5, 1, 0)
    Jtrain = np.mean(yhat != ytrain)
    Jtrains.append(Jtrain)
    yhat = net.predict(xdev)
    yhat = tf.math.sigmoid(yhat)
    yhat = np.where(yhat >= 0.5, 1, 0)
    Jdev = np.mean(yhat != ydev)
    Jdevs.append(Jdev)
for i in range(len(nets)):
    print(
        f"Network {i+1}: Training Set Classification Error: {Jtrains[i]:.5f}, "
        + f"CV Set Classification Error: {Jdevs[i]:.5f}"
    )


### GENERALIZATION ERROR ###
net = nets[np.argmin(Jdevs)]
yhat = net.predict(xtest)
yhat = tf.math.sigmoid(yhat)
yhat = np.where(yhat >= 0.5, 1, 0)
Jtest = np.mean(yhat != ytest)
print(f"Selected Network: {net.name}")
print(f"Test Set Classification Error: {Jtest:.5f}")