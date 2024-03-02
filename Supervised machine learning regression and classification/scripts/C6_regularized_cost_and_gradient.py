import numpy as np
from C1_sigmoid_function import sigmoid


# cost function regularized for linear regression
def cost_linear_regularized(X, y, w, b, lambda_=1):
    m = X.shape[0]
    f_wb = np.dot(X, w) + b
    cost = np.sum((f_wb - y) ** 2) / (2 * m)
    reg = np.sum(w**2) * (lambda_ / (2 * m))
    return cost + reg


# np.random.seed(1)
# X_tmp = np.random.rand(5, 6)
# y_tmp = np.array([0, 1, 0, 1, 0])
# w_tmp = (np.random.rand(X_tmp.shape[1]).reshape(-1,)- 0.5)
# b_tmp = 0.5
# lambda_tmp = 0.7
# cost_tmp = cost_linear_regularized(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
# print("Regularized cost:", cost_tmp)


# cost function regularized for logistic regression
def cost_logistic_regularized(X, y, w, b, lambda_=1):
    m = X.shape[0]
    z = X @ w + b
    g = sigmoid(z)
    j = np.sum(-y * np.log(g) - (1 - y) * np.log(1 - g)) / m
    reg = np.sum(w**2) * (lambda_ / (2 * m))
    return j + reg


# np.random.seed(1)
# X_tmp = np.random.rand(5,6)
# y_tmp = np.array([0,1,0,1,0])
# w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
# b_tmp = 0.5
# lambda_tmp = 0.7
# cost_tmp = cost_logistic_regularized(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
# print("Regularized cost:", cost_tmp)


# gradiant regularized for linear regression
def gradiant_regularized_linear(X, y, w, b, lambda_):
    m = X.shape[0]
    f_wb = X @ w + b
    e = f_wb - y
    dj_dw = (X.T @ e) / m
    dj_dw += w * (lambda_ / m)
    dj_db = np.sum(e / m)
    return dj_dw, dj_db


# np.random.seed(1)
# X_tmp = np.random.rand(5,3)
# y_tmp = np.array([0,1,0,1,0])
# w_tmp = np.random.rand(X_tmp.shape[1])
# b_tmp = 0.5
# lambda_tmp = 0.7
# dj_dw_tmp, dj_db_tmp =  gradiant_regularized_linear(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
# print(f"dj_db: {dj_db_tmp}", )
# print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )


# gradiant regularized for logistic regression
def gradiant_regularized_logistic(X, y, w, b, lambda_):
    m = X.shape[0]
    z = X @ w + b
    g = sigmoid(z)
    e = g - y
    dj_dw = (X.T @ e) / m
    dj_dw += w * (lambda_ / m)
    dj_db = np.sum(e / m)
    return dj_dw, dj_db


# np.random.seed(1)
# X_tmp = np.random.rand(5,3)
# y_tmp = np.array([0,1,0,1,0])
# w_tmp = np.random.rand(X_tmp.shape[1])
# b_tmp = 0.5
# lambda_tmp = 0.7
# dj_dw_tmp, dj_db_tmp =  gradiant_regularized_logistic(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
# print(f"dj_db: {dj_db_tmp}", )
# print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )