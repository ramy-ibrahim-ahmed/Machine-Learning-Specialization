import numpy as np
from C1_sigmoid_function import sigmoid
from C3_logistic_loss_and_cost_function import cost

X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])


def gradient(X, y, w, b):
    m = X.shape[0]
    f_wb = sigmoid((X @ w) + b)
    e = f_wb - y
    dj_dw = (1 / m) * X.T @ e
    dj_db = (1 / m) * np.sum(f_wb - y)
    return dj_dw, dj_db


# X_tmp = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
# y_tmp = np.array([0, 0, 0, 1, 1, 1])
# w_tmp = np.array([2.0, 3.0])
# b_tmp = 1.0
# dj_db_tmp, dj_dw_tmp = gradient(X_tmp, y_tmp, w_tmp, b_tmp)
# print(f"dj_db: {dj_db_tmp}")
# print(f"dj_dw: {dj_dw_tmp.tolist()}")


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


# w_tmp = np.zeros_like(X[0])
# b_tmp = 0.0
# alph = 0.1
# iters = 10000
# w_out, b_out = gradient_descent(X, y, w_tmp, b_tmp, alph, iters)
# print(f"\nupdated parameters: \nw:{w_out}, \nb:{b_out}")