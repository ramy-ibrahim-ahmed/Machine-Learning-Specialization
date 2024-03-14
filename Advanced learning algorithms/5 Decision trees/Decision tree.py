import numpy as np
import matplotlib.pyplot as plt

### DATASET ###
X = np.array(
    [
        [1, 1, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 0],
        [0, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]
)
y = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])


### ENTROPY ###
def entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


### PLOT ENTROPY ###
show = True
if show:
    p_arr = np.linspace(0, 1, 201)
    h = [entropy(p) for p in p_arr]
    plt.plot(p_arr, h, lw=1, label="entropy")
    plt.title("Entropy Equation", fontsize=16)
    plt.xlabel("P", fontsize=12)
    plt.ylabel("entropy", fontsize=12)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.grid(True, linestyle="--", lw=0.5)
    plt.show()


### SPLIT ON FEATURE ###
def Isplit(X, Ifeature):
    Ileft = []
    Iright = []
    for i, x in enumerate(X):
        if x[Ifeature] == 1:
            Ileft.append(i)
        else:
            Iright.append(i)
    return Ileft, Iright


### WEIGHTED ENTROPY ###
def entropyW(X, y, Ileft, Iright):
    Wleft = len(Ileft) / len(X)
    Wright = len(Iright) / len(X)
    Pplusl = np.sum(y[Ileft]) / len(Ileft)
    Pplusr = np.sum(y[Iright]) / len(Iright)
    return Wleft * entropy(Pplusl) + Wright * entropy(Pplusr)


Ileft, Iright = Isplit(X, 0)
print(entropyW(X, y, Ileft, Iright))


### INFORMATION GAIN ###
def IGain(X, y, Ileft, Iright):
    p_node = len(y[y == 1]) / len(y)
    h_node = entropy(p_node)
    return h_node - entropyW(X, y, Ileft, Iright)


print(IGain(X, y, Ileft, Ileft))


### COMPUTE INFORMATION GAIN FOR EVERY FEATURE ###
IGains = []
for i in range(X.shape[1]):
    Ileft, Iright = Isplit(X, i)
    IGains.append(IGain(X, y, Ileft, Iright))
    print(f"Feature: {i}, Information Gain: {IGains[i]:.2f}")