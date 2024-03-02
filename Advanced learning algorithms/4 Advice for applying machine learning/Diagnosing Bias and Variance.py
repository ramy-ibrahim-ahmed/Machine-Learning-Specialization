import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


### DATA SPLIT ###
def data_split(x, y, show=False):
    xtrain, xtemp, ytrain, ytemp = train_test_split(x, y, test_size=0.40, random_state=80)
    xdev, xtest, ydev, ytest = train_test_split(xtemp, ytemp, test_size=0.50, random_state=80)
    if show == True:
        plt.scatter(xtrain, ytrain, c="r", label="Train set")
        plt.scatter(xdev, ydev, label="Dev set")
        plt.scatter(xtest, ytest, c="y", label="Test set")
        plt.title(f"Data Splits", fontsize=16)
        plt.xlabel("X", fontsize=12)
        plt.ylabel("Y", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()
    return xtrain, xdev, xtest, ytrain, ydev, ytest


### POLYNOMIAL SELECTION ###
def evaluate_polys(model, xtrain, ytrain, xdev, ydev, max_degree, baseline, show=False):
    degrees = range(1, max_degree + 1)
    polys = []
    scalers = []
    models = []
    Jtrains = []
    Jdevs = []
    for degree in degrees:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        xtrain_ = poly.fit_transform(xtrain)
        polys.append(poly)
        scaler = StandardScaler()
        xtrain_ = scaler.fit_transform(xtrain_)
        scalers.append(scaler)
        model.fit(xtrain_, ytrain)
        models.append(model)
        yhat = model.predict(xtrain_)
        Jtrain = mean_squared_error(ytrain, yhat) / 2
        Jtrains.append(Jtrain)
        xdev_ = poly.transform(xdev)
        xdev_ = scaler.transform(xdev_)
        yhat = model.predict(xdev_)
        Jdev = mean_squared_error(ydev, yhat) / 2
        Jdevs.append(Jdev)
    if show:
        plt.plot(degrees, Jtrains, c="r", marker=".", label="MSE for Train")
        plt.plot(degrees, Jdevs, marker=".", label="MSE for Development")
        plt.plot(degrees, np.repeat(baseline, len(degrees)), c="black", linestyle="--", label="Baseline")
        plt.title(f"Polynomial Evaluation over {len(degrees)} degree", fontsize=16)
        plt.xlabel("Degree", fontsize=12)
        plt.ylabel("MSE", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()
    return polys, scalers, models, Jtrains, Jdevs


### LOW BIAS ###
data = np.loadtxt(r"data/c2w3_lab2_data1.csv", delimiter=",")
x = data[:, :-1]
y = data[:, -1:]
xtrain, xdev, xtest, ytrain, ydev, ytest = data_split(x, y)
model = LinearRegression()
max_degree = 10
baseline = 400
evaluate_polys(model, xtrain, ytrain, xdev, ydev, max_degree, baseline)


### HEIGH BIAS ###
data = np.loadtxt(r"data/c2w3_lab2_data1.csv", delimiter=",")
x = data[:, :-1]
y = data[:, -1:]
xtrain, xdev, xtest, ytrain, ydev, ytest = data_split(x, y)
model = LinearRegression()
max_degree = 10
baseline = 200
evaluate_polys(model, xtrain, ytrain, xdev, ydev, max_degree, baseline)


### GET ADDITIONAL FEATURES ###
### FIXING HIGH BIAS ###
data = np.loadtxt(r"data/c2w3_lab2_data2.csv", delimiter=",")
x = data[:, :-1]
y = data[:, -1:]
xtrain, xdev, xtest, ytrain, ydev, ytest = data_split(x, y)
model = LinearRegression()
max_degree = 6
baseline = 250
evaluate_polys(model, xtrain, ytrain, xdev, ydev, max_degree, baseline)


### REMOVE IRRELEVANT FEATURES ###
### FIXING HIGH VARIANCE ###
data = np.loadtxt(r"data/c2w3_lab2_data2.csv", delimiter=",")
x1 = data[:, :-1]
y1 = data[:, -1:]
xtrain1, xdev1, xtest1, ytrain1, ydev1, ytest1 = data_split(x1, y1)
data = np.loadtxt(r"data/c2w3_lab2_data3.csv", delimiter=",")
x2 = data[:, :-1]
y2 = data[:, -1:]
xtrain2, xdev2, xtest2, ytrain2, ydev2, ytest2 = data_split(x2, y2)
model = LinearRegression()
max_degree=4
baseline=250
_, _, _, Jtrains1, Jdevs1 = evaluate_polys(model, xtrain1, ytrain1, xdev1, ydev1, max_degree, baseline)
_, _, _, Jtrains2, Jdevs2 = evaluate_polys(model, xtrain2, ytrain2, xdev2, ydev2, max_degree, baseline)
degrees = range(1, max_degree + 1)
show = False
if show:
    plt.plot(degrees, Jtrains1, marker=".", c="b", linestyle="dotted", label="Train MSE 2 Features")
    plt.plot(degrees, Jdevs1, marker=".", c="b", linestyle="dotted", label="Dev MSE 2 Features")
    plt.plot(degrees, Jtrains2, marker=".", c="y", label="Train MSE 3 Features")
    plt.plot(degrees, Jdevs2, marker=".", c="y", label="Dev MSE 3 Features")
    plt.plot(degrees, np.repeat(baseline, len(degrees)), c="black", linestyle="--", label="Baseline")
    plt.title("Irrelevant Features on Polynomial Selection", fontsize=16)
    plt.xlabel("Degree", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


### REGULARIZATION PARAMETERS SELECTION ON A POLY DEGREE ###
def evaluate_ridgs(model, xtrain, ytrain, xdev, ydev, alphas, degree=1, baseline=None, show=False):
    models = []
    Jtrains = []
    Jdevs = []
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    xtrain = poly.fit_transform(xtrain)
    xdev = poly.transform(xdev)
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xdev = scaler.transform(xdev)
    for alpha in alphas:
        model.set_params(alpha=alpha)
        model.fit(xtrain, ytrain)
        models.append(model)
        yhat = model.predict(xtrain)
        Jtrain = mean_squared_error(ytrain, yhat) / 2
        Jtrains.append(Jtrain)
        yhat = model.predict(xdev)
        Jdev = mean_squared_error(ydev, yhat) / 2
        Jdevs.append(Jdev)
    if show:
        alphas = [str(x) for x in alphas]
        plt.plot(alphas, Jtrains, c='r', marker=".", label='Train MSE'); 
        plt.plot(alphas, Jdevs, marker=".", label='Dev MSE'); 
        plt.plot(alphas, np.repeat(baseline, len(alphas)), c="black", linestyle='--', label='Baseline')
        plt.title(f"Regularization Parameters on model degree = {degree}", fontsize=16)
        plt.xlabel("lambda", fontsize=12)
        plt.ylabel("MSE", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()
    return poly, scaler, models, Jtrains, Jdevs


### DECREASING THE REGULARIZATION PARAMETER ###
### FIXING HIGH BIAS ###
data = np.loadtxt(r"data/c2w3_lab2_data2.csv", delimiter=",")
x = data[:, :-1]
y = data[:, -1:]
xtrain, xdev, xtest, ytrain, ydev, ytest = data_split(x, y)
model = Ridge()
alphas = [10, 5, 2, 1, 0.5, 0.2, 0.1]
degree = 4
baseline = 250
evaluate_ridgs(model, xtrain, ytrain, xdev, ydev, alphas, degree, baseline)


### INCREASING THE REGULARIZATION PARAMETER ###
### FIXING HIGH VARIANCE ###
alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
evaluate_ridgs(model, xtrain, ytrain, xdev, ydev, alphas, degree, baseline)


### GET MORE DATA EXAMPLES ###
### FIXING HIGH VARIANCE ###
data = np.loadtxt(r"data/c2w3_lab2_data4.csv", delimiter=",")
x = data[:, :-1]
y = data[:, -1:]
xtrain, xdev, xtest, ytrain, ydev, ytest = data_split(x, y)
degree = 4
baseline = 250
poly = PolynomialFeatures(degree=degree, include_bias=False)
xtrain = poly.fit_transform(xtrain)
xdev = poly.transform(xdev)
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xdev = scaler.transform(xdev)
model = LinearRegression()
percents = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
Jtrains = []
Jdevs = []
samples = []
for percent in percents:
    Qtrain = len(xtrain) * percent // 100
    Qdev = len(xdev) * percent // 100
    samples.append(Qtrain + Qdev)
    xtrain_ = xtrain[:Qtrain]
    ytrain_ = ytrain[:Qtrain]
    xdev_ = xdev[:Qdev]
    ydev_ = ydev[:Qdev]
    model.fit(xtrain_, ytrain_)
    yhat = model.predict(xtrain_)
    Jtrain = mean_squared_error(ytrain_, yhat) / 2
    Jtrains.append(Jtrain)
    yhat = model.predict(xdev_)
    Jdev = mean_squared_error(ydev_, yhat) / 2
    Jdevs.append(Jdev)
show = False
if show:
    plt.plot(samples, Jtrains, c="r", marker=".", label="Train MSE")
    plt.plot(samples, Jdevs, marker=".", label="Dev MSE")
    plt.plot(samples, np.repeat(baseline, len(samples)), c="black", linestyle="--", label="Baseline")
    plt.title(f"Data Samples over Polynomial = {degree}", fontsize=16)
    plt.xlabel("Total Data Sample (train + dev)", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()