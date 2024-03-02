import numpy as np
import matplotlib.pyplot as plt
from script import *

x_train = np.array([0.0, 1, 2, 3, 4, 5])
y_train = np.array([0, 0, 0, 1, 1, 1])

pos = y_train == 1
neg = y_train == 0

# # plot
# plt.scatter(x_train[pos], y_train[pos], marker="x", s=80, c="red", label="y=1")
# plt.scatter(
#     x_train[neg],
#     y_train[neg],
#     marker="o",
#     s=100,
#     label="y=0",
#     facecolors="none",
#     edgecolors="blue",
#     lw=3,
# )
# plt.ylabel("y")
# plt.xlabel("x")
# plt.title("one variable plot")
# plt.legend()
# plt.show()

w, b, _ = gradient_descent(x_train.reshape(-1, 1), y_train, 0.1, 1000)
predictions = f_wb(x_train.reshape(-1, 1), w, b)

# # plot
# plt.plot(x_train, predictions, c="b")
# plt.scatter(x_train[pos], y_train[pos], marker="x", s=80, c="red", label="y=1")
# plt.scatter(
#     x_train[neg],
#     y_train[neg],
#     marker="o",
#     s=100,
#     label="y=0",
#     facecolors="none",
#     edgecolors="blue",
#     lw=3,
# )
# plt.ylabel("y")
# plt.xlabel("x")
# plt.title("one variable plot")
# plt.legend()
# plt.show()
