import numpy as np
import matplotlib.pyplot as plt

# features and targets
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# num of training examples
m = x_train.shape[0]
m = len(x_train)

# # plot data points
# plt.scatter(x_train, y_train, marker="x", c="r")
# plt.title("Housing Prices")
# plt.ylabel("Price (in 1000s of dollars)")
# plt.xlabel("Size (1000 sqft)")
# plt.show()

# weight and bias
w = 200
b = 100


# model function
def univariable_linear_regression(x, w, b):
    m = x.shape[0]
    f = np.zeros(m)
    for i in range(m):
        f[i] = w * x[i] + b
    return f


# train model
p = univariable_linear_regression(x_train, w, b)

# # plot model predictions
# plt.plot(x_train, p, c="b", label="Our Predictions")
# plt.scatter(x_train, y_train, marker="x", c="r", label="Acual Values")
# plt.title('House Prices')
# plt.ylabel('Price (in 1000s of dollars)')
# plt.xlabel('Size (1000 sqft)')
# plt.legend()
# plt.show()

# # predict new values after know the right w and b
# x_i = 1.2
# new_p = w * 1.2 + b
# print(f"{new_p:.0f},000$")


# cost function
def cost_function(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f = w * x[i] + b
        cost = (f - y[i]) ** 2
        cost_sum = cost_sum + cost
    j = (1 / (2 * m)) * cost_sum
    return j


# cost function with summision using numpy
def cost_function(x, y, w, b):
    cost = (1 / (2 * len(x))) * np.sum(((b + w * x) - y) ** 2)
    return cost


# # print cost of model
# print(cost_function(x_train, y_train, w, b))


# Gradient Descent
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


def gradient_descent(x, y, w_init, b_init, alpha, num_iters, gradient_fun):
    b = b_init
    w = w_init
    while num_iters:
        # new drivitive
        dj_dw, dj_db = gradient_fun(x, y, w, b)
        # new w & b
        w -= alpha * dj_dw
        b -= alpha * dj_db
        num_iters -= 1
    return w, b


# Gradient Descent with numpy sumision
def gradient_descent_sum(x, y, w_init, b_init, alpha, num_iters):
    m = x.shape[0]
    w_new, b_new = w_init, b_init
    for _ in range(num_iters):
        w = w_new
        b = b_new
        w_new -= alpha * np.sum(((w * x + b) - y) * x) / m
        b_new -= alpha * np.sum((w * x + b) - y) / m
    return w_new, b_new


# # run gradient descent
# w_init = 0
# b_init = 0
# iterations = 100000
# tmp_alpha = 1.0e-2
# w_final, b_final = gradient_descent_sum(
#     x_train,
#     y_train,
#     w_init,
#     b_init,
#     tmp_alpha,
#     iterations,
# )
# print(f"(w,b) found by gradient descent: ({w_final:.4f},{b_final:.4f})")

# # set large alpha
# iterations = 10
# tmp_alpha = 8.0e-1
# w_final, b_final = gradient_descent_sum(
#     x_train,
#     y_train,
#     w_init,
#     b_init,
#     tmp_alpha,
#     iterations,
# )
# print(f"(w,b) found by gradient descent: ({w_final:.4f},{b_final:.4f})")