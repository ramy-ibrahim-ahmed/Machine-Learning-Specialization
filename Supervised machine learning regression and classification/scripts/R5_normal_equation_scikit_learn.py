import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X_train = np.array([1.0, 2.0])
y_train = np.array([300, 500])

# Prepare features
X_train = X_train.reshape(-1, 1)

# Fit model
LR = LinearRegression()
LR.fit(X_train, y_train)

# View parameters
b = LR.intercept_
w = LR.coef_
print(f"w: {w.round(2)}", f"b: {b.round(2)}")

# Predict
X_test = np.array([[1200]])
y_pred = LR.predict(X_test)
print(f"Prediction for 1200 sqft house: ${y_pred[0]:0.2f}")