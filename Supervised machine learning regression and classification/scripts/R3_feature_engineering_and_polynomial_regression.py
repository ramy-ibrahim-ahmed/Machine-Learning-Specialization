import numpy as np
import matplotlib.pyplot as plt
import R1_multiple_linear_regression as B
import R2_feature_scaling_and_learning_rate as C

# Without feature engineering
x = np.arange(20)
y = 1 + x**2
X = x.reshape(-1, 1)

w, b, _ = B.gradient_descent(X, y, iterations=1000, alpha=1e-2)

# # Plot
# plt.scatter(x, y, marker="x", c="r", label="Actual Value")
# plt.plot(x, B.f_wb(X, w, b), label="Predicted Value")
# plt.title("no feature engineering")
# plt.xlabel("X")
# plt.ylabel("y")
# plt.legend()
# plt.show()

# With feature engineering
X = X**2

w, b, _ = B.gradient_descent(X, y, iterations=10000, alpha=1e-5)

# # Plot
# plt.scatter(x, y, marker="x", c="r", label="Actual values")
# plt.plot(x, B.f_wb(X, w, b), label="Prediction values")
# plt.title("Added x**2 feature")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()

# Selecting features
y = x**2
X = np.c_[x, x**2, x**3]

# w, b, _ = B.gradient_descent(X, y, iterations=10000, alpha=1e-7)

# plt.scatter(x, y, marker="x", c="r", label="Actual Value")
# plt.plot(x, B.f_wb(X, w, b), label="Predicted Value")
# plt.title("x, x**2, x**3 features")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()

# Notes which W's were changed from zero
# Thoes W's will be with selected features
# print(w.round(2), b.round(4))

# An Alternate View
# Selected features is the features that suits thr target
y = x**2
X = np.c_[x, x**2, x**3]
X_features = ["x", "x^2", "x^3"]

# fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
# for i in range(len(ax)):
#     ax[i].scatter(X[:, i], y)
#     ax[i].set_xlabel(X_features[i])
# ax[0].set_ylabel("y")
# plt.show()

# Scaling features
X = np.c_[x, x**2, x**3]
# print(np.ptp(X, axis=0), X)
# X = C.Z_score(X)
# print(np.ptp(X, axis=0), X)

# w, b, _ = B.gradient_descent(X, y, iterations=100000, alpha=1e-1)

# plt.scatter(x, y, marker="x", c="r", label="Actual Value")
# plt.title("Normalized x x**2, x**3 feature")
# plt.plot(x, B.f_wb(X, w, b), label="Predicted Value")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()

# Complex function
y = np.cos(x / 2)

X = np.c_[
    x,
    x**2,
    x**3,
    x**4,
    x**5,
    x**6,
    x**7,
    x**8,
    x**9,
    x**10,
    x**11,
    x**12,
    x**13,
]
# X = C.Z_score(X)

# w, b, _ = B.gradient_descent(X, y, iterations=1000000, alpha=1e-1)

# plt.scatter(x, y, marker="x", c="r", label="Actual Value")
# plt.plot(x, B.f_wb(X, w, b), label="Predicted Value")
# plt.title("Normalized x x**2, x**3 feature")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()