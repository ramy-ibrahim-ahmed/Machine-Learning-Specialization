import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

# train model
lr = LogisticRegression()
lr.fit(X, y)

# make prediction
y_pred = lr.predict(X)
print(f"Prediction on training set: {y_pred}")

# accuricy
print(f"Accuracy on training set: {lr.score(X, y) * 100}%")