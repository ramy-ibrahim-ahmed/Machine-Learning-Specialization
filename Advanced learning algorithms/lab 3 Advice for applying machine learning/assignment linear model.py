import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from public_tests_a1 import *
from assigment_utils import *


tf.keras.backend.set_floatx("float64")
tf.autograph.set_verbosity(0)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


### DATASET ###
# GENERATE DATA
X, y, x_ideal, y_ideal = gen_data(18, 2, 0.7)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
# SPLIT DATA TRAIN | TEST
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=1)
# PLOT
show = False
if show:
    plt.scatter(xtrain, ytrain, c="r", label="Train set")
    plt.scatter(xtest, ytest, label="Test set")
    plt.title("Data Splits", fontsize=16)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


### MEAN SQUARE ERROR FOR LINEAR REGRESSION ###
def mse(y, yhat):
    m = len(y)
    err = 0.0
    for i in range(m):
        loss = (yhat[i] - y[i]) ** 2
        err += loss
    err /= 2 * m
    return err


# TEST
y_hat = np.array([2.4, 4.2])
y_tmp = np.array([2.3, 4.1])
mse(y_hat, y_tmp)
test_eval_mse(mse)


### POLYNOMIAL FEATURES MODEL ###
degree = 10
poly = PolynomialFeatures(degree=degree, include_bias=False)
xtrain = poly.fit_transform(xtrain)
xtest = poly.transform(xtest)
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)
model = LinearRegression()
model.fit(xtrain, ytrain)
yhat = model.predict(xtrain)
Jtrain = mean_squared_error(ytrain, yhat) / 2
yhat = model.predict(xtest)
Jtest = mean_squared_error(ytest, yhat) / 2
print(Jtrain, Jtest)
# PLOT
show = False
if show:
    xmodel = np.linspace(0, int(X.max()), 1000).reshape(-1, 1)
    xmodel = poly.transform(xmodel)
    xmodel = scaler.transform(xmodel)
    ymodel = model.predict(xmodel).reshape(-1, 1)
    plt.scatter(xtrain[:, 0], ytrain, c="r", label="Train set")
    plt.scatter(xtest[:, 0], ytest, label="Test set")
    plt.plot(xmodel[:, 0], ymodel, linewidth=0.5, label=f"Model Function on degree = {degree}",)
    plt.title("Data Splits", fontsize=16)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


### DATA SPLIT TRAIN | DEV | TEST ###
# GENETARE DATA
X, y, x_ideal, y_ideal = gen_data(40, 5, 0.7)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
# SPLIT DATA
xtrain, xtemp, ytrain, ytemp = train_test_split(X, y, test_size=0.40, random_state=1)
xdev, xtest, ydev, ytest = train_test_split(xtemp, ytemp, test_size=0.50, random_state=1)
# PLPT
show = False
if show:
    plt.scatter(xtrain, ytrain, c="r", label="Train set")
    plt.scatter(xdev, ydev, label="Development set")
    plt.scatter(xtest, ytest, c="g", label="Test set")
    plt.title("Data Splits", fontsize=16)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


### TRY POLYNOMIAL FEATURES MODELS ###
Jtrains = []
Jdevs = []
polys = []
scalers = []
ymodels = np.empty((1000, 9))
xmodel = np.linspace(0, int(X.max()), 1000).reshape(-1, 1)
degrees = range(1, 10)
model = LinearRegression()
for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    xtrain_ = poly.fit_transform(xtrain)
    xdev_ = poly.transform(xdev)
    xmodel_ = poly.transform(xmodel)
    polys.append(poly)
    scaler = StandardScaler()
    xtrain_ = scaler.fit_transform(xtrain_)
    xdev_ = scaler.transform(xdev_)
    xmodel_ = scaler.transform(xmodel_)
    scalers.append(scaler)
    model.fit(xtrain_, ytrain)
    yhat = model.predict(xtrain_)
    Jtrain = mean_squared_error(ytrain, yhat) / 2
    Jtrains.append(Jtrain)
    yhat = model.predict(xdev_)
    Jdev = mean_squared_error(ydev, yhat) / 2
    Jdevs.append(Jdev)
    ymodel = model.predict(xmodel_).reshape(-1,)
    ymodels[:, degree - 1] = ymodel
Odegree = np.argmin(Jdevs) + 1
# PLOT
show = False
if show:
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].set_title("Models Representation", fontsize=14)
    ax[0].set_xlabel("X", fontsize=12)
    ax[0].set_ylabel("Y", fontsize=12)
    ax[0].scatter(xtrain, ytrain, c="r", label="Train set")
    ax[0].scatter(xdev, ydev, label="Development set")
    for degree in degrees:
        ax[0].plot(xmodel, ymodels[:, (degree - 1)], linewidth=0.5, label=f"degree = {degree}")
    ax[0].legend(loc="upper left")
    ax[0].grid(True, linestyle="--", linewidth=0.5)
    ax[1].set_title("MSE on Degrees", fontsize=14)
    ax[1].set_xlabel("Degree", fontsize=12)
    ax[1].set_ylabel("MSE", fontsize=12)
    ax[1].plot(degrees, Jtrains, c="r", marker=".", lw=2, label="Train MSE")
    ax[1].plot(degrees, Jdevs, marker=".", lw=2, label="Development MSE")
    ax[1].axvline(Odegree, linewidth=1, c="black", linestyle="--")
    ax[1].legend()
    ax[1].grid(True, linestyle="--", linewidth=0.5)
    fig.suptitle("Find Optimal Polunominal Degree",fontsize = 16)
    plt.tight_layout()
    plt.show()


### TRY REGULARIZATION PARAMETERS ###
alphas = np.array([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
degree = 10
Jtrains = []
Jdevs = []
models = []
xmodel = np.linspace(0, int(X.max()), 1000).reshape(-1, 1)
ymodel = np.zeros(shape=(1000, len(alphas)))
poly = PolynomialFeatures(degree=degree, include_bias=False)
xtrain_ = poly.fit_transform(xtrain)
xdev_ = poly.transform(xdev)
xmodel_ = poly.transform(xmodel)
scaler = StandardScaler()
xtrain_ = scaler.fit_transform(xtrain_)
xdev_ = scaler.transform(xdev_)
xmodel_ = scaler.transform(xmodel_)
model = Ridge()
for alpha in alphas:
    model.set_params(alpha=alpha)
    model.fit(xtrain_, ytrain)
    models.append(model)
    yhat = model.predict(xtrain_)
    Jtrain = mean_squared_error(ytrain, yhat) / 2
    Jtrains.append(Jtrain)
    yhat = model.predict(xdev_)
    jdev = mean_squared_error(ydev, yhat) / 2
    Jdevs.append(jdev)
    yhat = model.predict(xmodel_)
    ymodel[:, np.where(alphas == alpha)[0]] = yhat
Omodel = models[np.argmin(Jdevs)]
# PLOT
show = False
if show:
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].set_title("Models Representation", fontsize=14)
    ax[0].set_xlabel("X", fontsize=12)
    ax[0].set_ylabel("Y", fontsize=12)
    ax[0].scatter(xtrain, ytrain, c="r", marker=".", label="Train set")
    ax[0].scatter(xdev, ydev, marker=".", label="Development set")
    for alpha in (0, 3, 7, 9):
        ax[0].plot(xmodel, ymodel[:, alpha], lw=0.5, label=f"lambda = {alphas[alpha]}")
    ax[0].legend(loc="upper left")
    ax[0].grid(True, linestyle="--", lw=0.5)
    ax[1].set_title("MSE on Lambdas", fontsize=14)
    ax[1].set_xlabel("Lambda", fontsize=12)
    ax[1].set_ylabel("MSE", fontsize=12)
    ax[1].set_xscale('log')
    ax[1].plot(alphas, Jtrains, c="r", lw=2, marker=".", label="Train MSE")
    ax[1].plot(alphas, Jdevs, lw=2, marker=".", label="Development MSE")
    ax[1].axvline(alphas[np.argmin(Jdevs)], lw=1, c="black", linestyle="--")
    ax[1].legend()
    ax[1].grid(True, linestyle="--", lw=0.5)
    ax[1].text(0.05 ,0.44 ,"High\nVariance",fontsize=12, ha='left', transform=ax[1].transAxes, c="black")
    ax[1].text(0.95 ,0.44 ,"High\nBias" ,fontsize=12, ha='right', transform=ax[1].transAxes, c="black")
    ax[1].legend(loc='upper left')
    fig.suptitle("Tuning Regularization Parameter", fontsize=16)
    plt.tight_layout()
    plt.show()


### INCREASE DATASET ###
samples = np.array(50 * np.arange(1,16))
Jtrains = []
Jdevs = []
ymodel = np.zeros((100, len(samples)))
model = LinearRegression()
for i in range(len(samples)):
    X, y, _, _ = gen_data(samples[i], 5, 0.7)
    xmodel = np.linspace(0, int(X.max()), 100)  
    xtrain, X_, ytrain, y_ = train_test_split(X, y, test_size=0.40, random_state=1)
    xdev, xtest, ydev, ytest = train_test_split(X_, y_, test_size=0.50, random_state=1)
    poly = PolynomialFeatures(degree=16, include_bias=False)
    xtrain_ = poly.fit_transform(xtrain.reshape(-1, 1))
    xdev_ = poly.transform(xdev.reshape(-1, 1))
    xmodel_ = poly.transform(xmodel.reshape(-1, 1))
    scaler = StandardScaler()
    xtrain_ = scaler.fit_transform(xtrain_)
    xdev_ = scaler.transform(xdev_)
    xmodel_ = scaler.transform(xmodel_)
    model.fit(xtrain_, ytrain)
    yhat = model.predict(xtrain_)
    Jtrain = mean_squared_error(ytrain, yhat) / 2
    Jtrains.append(Jtrain)
    yhat = model.predict(xdev_)
    Jdev = mean_squared_error(ydev, yhat) / 2
    Jdevs.append(Jdev)
    ymodel[:, i] = model.predict(xmodel_)
# PLOT
show = False
if show:
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].set_title("Model Representation",fontsize = 14)
    ax[0].set_xlabel("X", fontsize=12)
    ax[0].set_ylabel("Y", fontsize=12)
    ax[0].set_ylim(0, 3000)
    ax[0].scatter(np.concatenate([xtrain, xdev]), np.concatenate([ytrain, ydev]), c="orange", s=10, marker=".", label="Data set")
    for i in range(0, len(samples), 3):
        ax[0].plot(xmodel, ymodel[:, i], lw=1, label=f"m = {samples[i]}")
    ax[0].legend(loc='upper left')
    ax[1].set_title("MSE on data samples",fontsize = 14)
    ax[1].plot(samples, Jtrains, c="r", marker=".", lw=2, label="Train MSE")
    ax[1].plot(samples, Jdevs, marker=".", lw=2, label="Development MSE")
    ax[1].set_xlabel("Number of examples (m)", fontsize=12)
    ax[1].set_ylabel("MSE", fontsize=12)
    fig.suptitle("Tuning number of examples",fontsize = 16)
    ax[1].text(0.05, 0.5, "High\nVariance", fontsize=12, ha='left', transform=ax[1].transAxes,c="black")
    ax[1].text(0.95, 0.5, "Good \nGeneralization", fontsize=12, ha='right', transform=ax[1].transAxes, c="black")
    ax[1].legend()
    plt.tight_layout()
    plt.show()