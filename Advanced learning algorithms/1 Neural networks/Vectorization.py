import numpy as np


### VECTOR @ VECTOR ###
### WHEN DEAL WITH VECTORS USE DOT ###
x = np.array(
    [
        [1],
        [2],
    ]
)  # (2, 1)
w = np.array(
    [
        [3],
        [4],
    ]
)  # (2, 1)
xt = x.T  # (1, 2)
print(np.dot(xt, w))  # (1, 2) @ (2, 1)
xt @ w


### VECTOR @ MATRIX ###
### WHEN DEAL WITH MATRICES USE MATMUL ###
W = np.array(
    [
        [3, 5],
        [4, 6],
    ]
)  # (2, 2)
z = np.matmul(xt, w)  # (1, 2) @ (2, 2)
print(z)  # xt @ w


### MATRIX @ MATRIX ###
### WHEN DEALING WITH MATRICES USE MATMUL ###
A = np.array(
    [
        [1, -1, 0.1],
        [2, -2, 0.2],
    ]
)  # (2, 3)
W = np.array(
    [
        [3, 5, 7, 9],
        [4, 6, 8, 0],
    ]
)  # (2, 4)
At = A.T  # (3, 2)
print(np.matmul(At, W))  # (3, 2) @ (2, 4)
print(np.dot(At, W))
print(At @ W)