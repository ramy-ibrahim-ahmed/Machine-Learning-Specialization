import numpy as np
import matplotlib.pyplot as plt
from C1_sigmoid_function import sigmoid
from C2_decision_boundary import plot_data

X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# # plot
# fig, ax = plt.subplots(1, 1)
# plot_data(X_train, y_train, ax)
# ax.axis([0, 4, 0, 3.5])
# ax.set_ylabel("$x_1$", fontsize=12)
# ax.set_xlabel("$x_0$", fontsize=12)
# plt.show()


# cost function
def cost(X, y, w, b):
    m = X.shape[0]
    z = (X @ w) + b
    g = sigmoid(z)
    loss = np.sum(y * np.log(g) + (1 - y) * np.log(1 - g))
    return -loss / m


# use cost function
w_tmp = np.array([1, 1])
b_tmp = -3
print(cost(X_train, y_train, w_tmp, b_tmp))

# # try defferent w value
# x0 = np.arange(0, 6)
# x1 = 3 - x0
# x1_other = 4 - x0
# fig, ax = plt.subplots(1, 1, figsize=(4, 4))
# ax.plot(x0, x1, c="b", label="$b$=-3")
# ax.plot(x0, x1_other, c="y", label="$b$=-4")
# ax.axis([0, 4, 0, 4])
# plot_data(X_train, y_train, ax)
# ax.axis([0, 4, 0, 4])
# ax.set_ylabel("$x_1$", fontsize=12)
# ax.set_xlabel("$x_0$", fontsize=12)
# plt.legend(loc="upper right")
# plt.title("Decision Boundary")
# plt.show()

# compare costs
w_array1 = np.array([1, 1])
b_1 = -3
w_array2 = np.array([1, 1])
b_2 = -4
print("Cost for b = -3 : ", cost(X_train, y_train, w_array1, b_1))
print("Cost for b = -4 : ", cost(X_train, y_train, w_array2, b_2))
