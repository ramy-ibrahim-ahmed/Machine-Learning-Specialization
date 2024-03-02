import numpy as np


# features & labels
x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])


# model function using loop
def f_wb_loop(x, w, b):
    m = x.shape[0]
    prediction = 0
    for i in range(m):
        p_i = x[i] * w[i]
        prediction += p_i
    prediction += b
    return prediction


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


# # get a row from dataset
# vector = x_train[0, :]

# # parameters
# b_init = 785.1811367994083
# w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

# # run model
# prediction = f_wb(vector, w_init, b_init)
# prediction_loop = f_wb_loop(vector, w_init, b_init)
# cost = j_wb(x_train, y_train, w_init, b_init)

# print(f"prediction using numpy: {prediction.round(2)}")
# print(f"prediction using loop: {prediction_loop.round(2)}")
# print(f"cost: {cost}")


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

# # initialization
# initial_w = np.zeros_like(w_init)
# initial_b = 0.0
# iterations = 1000
# alpha = 5.0e-7
# # run gradient descent
# w_final, b_final, Js = gradient_descent(
#     x_train,
#     y_train,
#     alpha=alpha,
#     iterations=iterations,
# )
# print(f"b,w found by gradient descent: {b_final.round(2)},{w_final.round(2)}")
# for i in range(x_train.shape[0]):
#     print(f"prediction: {np.dot(x_train[i], w_final) + b_final}, target: {y_train[i]}")