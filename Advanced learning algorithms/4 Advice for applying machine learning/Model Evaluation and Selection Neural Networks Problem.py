import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


os.environ["CUDA_VISIBLE_DEVICES"] = (r"PCI\VEN_10DE&DEV_1F95&SUBSYS_3FA317AA&REV_A1\4&11F4856E&0&0008")
np.set_printoptions(precision=2)
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)
tf.random.set_seed(1234)
np.random.seed(1234)


### DATASET ###
data = np.loadtxt(r"./data/data_w3_ex1.csv", delimiter=",")
x = data[:, 0]
y = data[:, 1]
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)


### PLOT DATA ###
# plt.scatter(x, y, s=20, label="Data")
# plt.xlabel("Feature", fontsize=12)
# plt.ylabel("Label", fontsize=12)
# plt.title("Data with Targer", fontsize=14)
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.show()


### DATA SPLIT ###
xtrain, xtemp, ytrain, ytemp = train_test_split(x, y, test_size=0.40, random_state=1)
xdev, xtest, ydev, ytest = train_test_split(xtemp, ytemp, test_size=0.50, random_state=1)
del xtemp, ytemp


### PLOT SPLITS ###
# plt.scatter(xtrain, ytrain, s=20, label="Train")
# plt.scatter(xdev, ydev, s=20, c="r", label="Development")
# plt.scatter(xtest, ytest, s=20, c="y", label="Test")
# plt.xlabel("Y", fontsize=12)
# plt.ylabel("Y", fontsize=12)
# plt.title("Data Splits", fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.show()


### FEATURE SCALING ###
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xdev = scaler.transform(xdev)
xtest = scaler.transform(xtest)


### FEED POLYNOMIAL FEATURES IN NETWORK ###
# FROM MODEL SELECTION WE GOT 4 AS THE BEST DEGREE 
# degree = 4
# poly = PolynomialFeatures(degree=degree, include_bias=False)
# xtrain = poly.fit_transform(xtrain)
# xdev = poly.transform(xdev)
# xtest = poly.transform(xtest)


### BUILD NETWORKS ###
degree = 1
network1 = Sequential(
    [
        Input(shape=(degree,)),
        Dense(units=25, activation=relu),
        Dense(units=15, activation=relu),
        Dense(units=1, activation=linear),
    ],
    name="network1",
)
network2 = Sequential(
    [
        Input(shape=(degree,)),
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
        Input(shape=(degree,)),
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
        loss="mse",
        optimizer=Adam(learning_rate=0.1),
    )
    net.fit(xtrain, ytrain, epochs=300, verbose=0)
    yhat = net.predict(xtrain)
    Jtrain = mean_squared_error(ytrain, yhat) / 2
    Jtrains.append(Jtrain)
    yhat = net.predict(xdev)
    Jdev = mean_squared_error(ydev, yhat) / 2
    Jdevs.append(Jdev)
print("RESULTS:")
for i in range(len(nets)):
    print(f"Network {i+1}: Training MSE: {Jtrains[i]:.2f}, " + f"Dev MSE: {Jdevs[i]:.2f}")


### GENERALIZATION ERROR ###
net = nets[np.argmin(Jdevs)]
yhat = net.predict(xtest)
Jtest = mean_squared_error(ytest, yhat) / 2
print(f"Selected Network: {net.name}")
print(f"Test MSE: {Jtest:.2f}")