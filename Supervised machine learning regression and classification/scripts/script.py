######################################### Linear Regression #
import numpy as np


# model function with numpy
def f_wb(x, w, b):
    prediction = np.dot(x, w) + b
    return prediction


# cost function
def j_wb(x, y, w, b):
    m = x.shape[0]
    j = np.sum(((np.dot(x, w) + b) - y) ** 2)
    j /= 2 * m
    return j


# Gradient
def gradient(x, y, w, b):
    m = x.shape[0]
    f_wb = x @ w + b
    e = f_wb - y
    dj_dw = (1 / m) * (x.T @ e)
    dj_db = (1 / m) * np.sum(e)
    return dj_db, dj_dw


# Gradient Descent
def gradient_descent(X, y, alpha, iterations, j_wb=j_wb, gradient=gradient):
    w = np.zeros(X.shape[1])
    b = 0
    Js = []
    for i in range(iterations):
        dj_db, dj_dw = gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        Js.append(j_wb(X, y, w, b))
        if iterations > 0 and i % (iterations // 10) == 0:
            print("Iteration {:4d}: Cost {:8.2f}".format(i, Js[-1]))
    return w, b, Js


# Feature Scaling
def Z_score(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm


####################################### Logistic Regression #
# sigmoid equassion
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


# model function with numpy
def g(X, w, b):
    z = X @ w + b
    prediction = sigmoid(z)
    return prediction


# cost function
def cost(X, y, w, b):
    m = X.shape[0]
    z = (X @ w) + b
    g = sigmoid(z)
    loss = np.sum(y * np.log(g) + (1 - y) * np.log(1 - g))
    return -loss / m


def gradient(X, y, w, b):
    m = X.shape[0]
    f_wb = sigmoid((X @ w) + b)
    e = f_wb - y
    dj_dw = (1 / m) * X.T @ e
    dj_db = (1 / m) * np.sum(f_wb - y)
    return dj_dw, dj_db


def gradient_descent(X, y, w, b, alpha, num_iters):
    J = []
    for i in range(num_iters):
        dj_dw, dj_db = gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        J.append(cost(X, y, w, b))
        if num_iters > 0 and i % (num_iters // 10) == 0:
            print("Iteration {:4d}: Cost {:8.2f}".format(i, J[-1]))
    return w, b


############################################ Regularization #
def cost_linear_regularized(X, y, w, b, lambda_=1):
    m = X.shape[0]
    f_wb = np.dot(X, w) + b
    j = np.sum((f_wb - y) ** 2) / (2 * m)
    reg = np.sum(w**2) * (lambda_ / (2 * m))
    return j + reg


def cost_logistic_regularized(X, y, w, b, lambda_=1):
    m = X.shape[0]
    z = X @ w + b
    g = sigmoid(z)
    j = np.sum(-y * np.log(g) - (1 - y) * np.log(1 - g)) / m
    reg = np.sum(w**2) * (lambda_ / (2 * m))
    return j + reg


def gradiant_regularized_linear(X, y, w, b, lambda_):
    m = X.shape[0]
    f_wb = X @ w + b
    e = f_wb - y
    dj_dw = (X.T @ e) / m
    dj_dw += w * (lambda_ / m)
    dj_db = np.sum(e / m)
    return dj_dw, dj_db


def gradiant_regularized_logistic(X, y, w, b, lambda_):
    m = X.shape[0]
    z = X @ w + b
    g = sigmoid(z)
    e = g - y
    dj_dw = (X.T @ e) / m
    dj_dw += w * (lambda_ / m)
    dj_db = np.sum(e / m)
    return dj_dw, dj_db