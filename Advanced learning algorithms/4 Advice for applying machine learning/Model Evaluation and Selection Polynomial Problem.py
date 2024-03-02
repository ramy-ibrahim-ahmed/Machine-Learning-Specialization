import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


### DATASET ###
data = np.loadtxt(r"./data/data_w3_ex1.csv", delimiter=",")
x = data[:, 0]
y = data[:, 1]
# 1D ARRAY -> 2D ARRAY
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)
print(x.shape)


### DATA VISUALIZATION ###
# plt.scatter(x, y, s=20, label="Data")
# plt.xlabel("Feature", fontsize=12)
# plt.ylabel("Label", fontsize=12)
# plt.title("Data with Targer", fontsize=14)
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.show()


### DATA SPLIT ###
# TRAIN 60 | DEV 20 | TEST 20
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.40, random_state=1)
x_dev, x_test, y_dev, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=1)
del x_temp, y_temp
print(x.shape, x_train.shape, x_dev.shape, x_test.shape)
# VISUALIZATION
# plt.scatter(x_train, y_train, s=20, label="Train")
# plt.scatter(x_dev, y_dev, s=20, c="r", label="Cross-validation")
# plt.scatter(x_test, y_test, s=20, c="y", label="Test")
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.legend(fontsize=12)
# plt.xlabel("X",fontsize=12)
# plt.ylabel("Y",fontsize=12)
# plt.show()


### FEATURE SCALING ###
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_train_mean = scaler.mean_.squeeze()
x_train_std = scaler.scale_.squeeze()
# DATA WITH MEAN = 0 | STD = 1
print(f"x_train mean: {x_train_mean:.02f}")
print(f"x_train scaled mean: {np.mean(x_train_scaled):.02f}")
print(f"x_train standar division: {x_train_std:.02f}")
print(f"x_train scaled standar division: {np.std(x_train_scaled):.02f}")
# VISUALIZATION
# plt.scatter(x_train_scaled, y_train, s=20, label="Data")
# plt.xlabel("Feature", fontsize=12)
# plt.ylabel("Label", fontsize=12)
# plt.title("Data After Scaled by Z-score", fontsize=14)
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.show()


### MODEL TRAINING ###
linear_model = LinearRegression()
linear_model.fit(x_train_scaled, y_train)


### MODEL EVALUATION ###
yhat_train = linear_model.predict(x_train_scaled)
J_train = mean_squared_error(y_train, yhat_train) / 2  # --> divide by 2 for the convention
print(f"Training MSE = {J_train}")
# SCALE THE DEV SET TO COMPUTE J ON DEV SET
# SCALE WITH THE SAME MEAN & STD (SCALER) OF TRAIN SET
# TRANSFORM ONLY NOT FIT TRANSFORM
x_dev_scaled = scaler.transform(x_dev)
yhat_dev = linear_model.predict(x_dev_scaled)
J_dev = mean_squared_error(y_dev, yhat_dev) / 2
print(f"Cross Validation MSE = {J_dev}")


### POLYNOMIAL FEATURES ###
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_mapped = poly.fit_transform(x_train)
print(x_train[:5])
print(x_train_mapped[:5])


### SCALE POLYNOMIAL ###
scaler_poly = StandardScaler()
x_train_mapped_scaled = scaler_poly.fit_transform(x_train_mapped)
print(np.ptp(x_train_mapped, axis=0))
print(np.ptp(x_train_mapped_scaled, axis=0))


### TRAIN POLYNOMIAL ###
model_poly = LinearRegression()
model_poly.fit(x_train_mapped_scaled, y_train)


### TRAIN EVALUATION ###
yhat_poly_train = model_poly.predict(x_train_mapped_scaled)
Jpoly_train = mean_squared_error(y_train, yhat_poly_train) / 2
print(f"Cost Train Polynominal: {Jpoly_train}")


### DEVELOPMENT EVALUATION ###
x_dev_mapped = poly.transform(x_dev)
x_dev_mapped_scaled = scaler_poly.transform(x_dev_mapped)
yhat_poly_dev = model_poly.predict(x_dev_mapped_scaled)
Jpoly_dev = mean_squared_error(y_dev, yhat_poly_dev) / 2
print(f"Cost dev Polynominal: {Jpoly_dev}")


### ITERATE OVER MANY POLYNOMIAL MODELS ###
J_trains = []
J_devs = []
models = []
polys = []
scalers = []
for i in range(1, 11):
    poly = PolynomialFeatures(degree=i, include_bias=False)
    polys.append(poly)
    xtrain = poly.fit_transform(x_train)
    scaler = StandardScaler()
    scalers.append(scaler)
    xtrain = scaler.fit_transform(xtrain)
    model = LinearRegression()
    model.fit(xtrain, y_train)
    models.append(model)
    yhat = model.predict(xtrain)
    Jtrain = mean_squared_error(y_train, yhat) / 2
    J_trains.append(Jtrain)
    xdev = poly.transform(x_dev)
    xdev = scaler.transform(xdev)
    yhat = model.predict(xdev)
    Jdev = mean_squared_error(y_dev, yhat) / 2
    J_devs.append(Jdev)
# degrees = range(1,11)
# plt.plot(degrees, J_trains, marker='o', label='Training MESs'); 
# plt.plot(degrees, J_devs, c='r', marker='o', label='Development MSEs')
# plt.title("Models MSEs", fontsize=16)
# plt.xlabel("Polynomial degree", fontsize=12)
# plt.ylabel("MSE", fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.show()


### CHOOSE BEST MODEL BASED ON BEST DEV MSE ###
degree = np.argmin(J_devs)
print(f"Lowest Dev MSE is found in the model with degree = {degree + 1}")


### GENERALIZATION ERROR ###
xtest_poly = polys[degree].transform(x_test)
xtest_poly = scalers[degree].transform(xtest_poly)
yhat = models[degree].predict(xtest_poly)
Jtest = mean_squared_error(y_test, yhat) / 2
print(f"Training MSE: {J_trains[degree]:.2f}")
print(f"DEVELOPMENT MSE: {J_devs[degree]:.2f}")
print(f"Test MSE: {Jtest:.2f}")