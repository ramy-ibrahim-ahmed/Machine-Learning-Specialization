import numpy as np
import matplotlib.pyplot as plt
from script import *

# e, exp
e = np.e
exp_vector = np.exp([1, 2, 3])
exp_scaler = np.exp(1)
print(e, exp_scaler, exp_vector, sep="\n")


# sigmoid equassion
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


# try sigmoid in range -10 -> 10
x = np.arange(-10, 11)
y = sigmoid(x).round(4)
print("Input (x), Output (sigmoid(x))")
print(np.c_[x, y].round(2))


# # plot
# fig, ax = plt.subplots(1, 1, figsize=(5, 3))
# ax.plot(x, y, c="b")
# ax.set_title("Sigmoid function")
# ax.set_ylabel("sigmoid(x)")
# ax.set_xlabel("x")
# plt.show()