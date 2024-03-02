import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)

# plot data
def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)
    neg = neg.reshape(-1,)
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, label=neg_label, facecolors='none', edgecolors='blue', lw=3)
    ax.legend(loc=loc)


# # plot
# fig,ax = plt.subplots(1,1,figsize=(4,4))
# plot_data(X, y, ax)
# ax.axis([0, 4, 0, 3.5])
# ax.set_ylabel('$x_1$')
# ax.set_xlabel('$x_0$')
# plt.show()


# # hyperparameters
# # z = x0 + x1 -3
# # on decision boundary z = 0
# # x0 + x1 = 3
# x0 = np.arange(0,6)
# x1 = 3 - x0
# fig,ax = plt.subplots(1,1,figsize=(5,4))
# # Plot the decision boundary
# ax.plot(x0,x1, c="b")
# ax.axis([0, 4, 0, 3.5])
# # Fill the region below the line
# ax.fill_between(x0,x1, alpha=0.2)
# # Plot the original data
# plot_data(X,y,ax)
# ax.set_ylabel(r'$x_1$')
# ax.set_xlabel(r'$x_0$')
# plt.show()