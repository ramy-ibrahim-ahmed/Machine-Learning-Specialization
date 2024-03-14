import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

RANDOM_STATE = 55

### LOAD DATASET ###
df = pd.read_csv("heart.csv")
print(df.head())


### ONE-HOT ENCODING ###
catigorical_vars = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
df = pd.get_dummies(data=df, prefix=catigorical_vars, columns=catigorical_vars)
print(df.head())


### SEPARATE TARGET COLUMN ###
print(len(df.columns))
Itrain = [x for x in df.columns if x != "HeartDisease"]
print(len(Itrain))


### SPLIT DATA ###
xtrain, xtest, ytrain, ytest = train_test_split(
    df[Itrain], df["HeartDisease"], train_size=0.8, random_state=RANDOM_STATE
)
print(f"Feature train samples = {len(xtrain)}, Feature test sampels = {len(xtest)}")
print(f"Target train samples = {len(ytrain)}, Target test sampels = {len(ytest)}")


### TREE BUILD ###
### VALIDATE MIN NUM OF SAMPLES TO SPLIT ###
min_samples_splits = [2, 10, 30, 50, 100, 200, 300, 700]
Jtrains = []
Jtests = []
for min_split in min_samples_splits:
    model = DecisionTreeClassifier(
        min_samples_split=min_split, random_state=RANDOM_STATE
    )
    model.fit(xtrain, ytrain)
    yhat = model.predict(xtrain)
    Jtrain = accuracy_score(yhat, ytrain)
    Jtrains.append(Jtrain)
    yhat = model.predict(xtest)
    Jtest = accuracy_score(yhat, ytest)
    Jtests.append(Jtest)
SHOW = False
if SHOW:
    plt.plot(Jtrains, label="Train accuracy")
    plt.plot(Jtests, c="r", label="Test accuracy")
    plt.xticks(ticks=range(len(min_samples_splits)), labels=min_samples_splits)
    plt.xlabel("min samples to split", fontsize=12)
    plt.ylabel("accuracy", fontsize=12)
    plt.title("Accuracy on min num of samples to split", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", lw=0.5)
    plt.tight_layout()
    plt.show()


### VALIDATE MAX TREE DEPTH ###
max_depths = [1, 2, 3, 4, 8, 16, 32, 64, None]
Jtrains = []
Jtests = []
for max_depth in max_depths:
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    model.fit(xtrain, ytrain)
    yhat = model.predict(xtrain)
    Jtrain = accuracy_score(yhat, ytrain)
    Jtrains.append(Jtrain)
    yhat = model.predict(xtest)
    Jtest = accuracy_score(yhat, ytest)
    Jtests.append(Jtest)
SHOW = False
if SHOW:
    plt.plot(Jtrains, label="Train accuracy")
    plt.plot(Jtests, c="r", label="Test accuracy")
    plt.xticks(ticks=range(len(max_depths)), labels=max_depths)
    plt.xlabel("max tree depth", fontsize=12)
    plt.ylabel("accuracy", fontsize=12)
    plt.title("Accuracy on max tree depth", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", lw=0.5)
    plt.tight_layout()
    plt.show()


### DECISION MAKING ###
max_depth = 4
min_samples_split = 50
tree = DecisionTreeClassifier(
    min_samples_split=min_samples_split, max_depth=max_depth, random_state=RANDOM_STATE
)
tree.fit(xtrain, ytrain)
Jtrain = accuracy_score(tree.predict(xtrain), ytrain)
Jtest = accuracy_score(tree.predict(xtest), ytest)
print(f"score in train (tree) = {Jtrain:.4f}")
print(f"score in test (tree) = {Jtest:.4f}")


### RANDOM FOREST ###
### VALIDATE MIN NUM OF SAMPLES TO SPLIT ###
min_samples_splits = [2, 10, 30, 50, 100, 200, 300, 700]
Jtrains = []
Jtests = []
for split in min_samples_splits:
    forest = RandomForestClassifier(min_samples_split=split, random_state=RANDOM_STATE)
    forest.fit(xtrain, ytrain)
    Jtrain = forest.predict(xtrain)
    Jtrains.append(accuracy_score(Jtrain, ytrain))
    Jtest = forest.predict(xtest)
    Jtests.append(accuracy_score(Jtest, ytest))
SHOW = False
if SHOW:
    plt.plot(Jtrains, label="Train accuracy")
    plt.plot(Jtests, c="r", label="Test accuracy")
    plt.xticks(ticks=range(len(min_samples_splits)), labels=min_samples_splits)
    plt.xlabel("min samples to splite", fontsize=12)
    plt.ylabel("accuracy", fontsize=12)
    plt.title("Accuracy on min samples to split", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", lw=0.5)
    plt.tight_layout()
    plt.show()


### VALIDATE MAX TREE DEPTH ###
max_depths = [2, 4, 8, 16, 32, 64, None]
Jtrains = []
Jtests = []
for max_depth in max_depths:
    forest = RandomForestClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    forest.fit(xtrain, ytrain)
    Jtrain = forest.predict(xtrain)
    Jtrains.append(accuracy_score(Jtrain, ytrain))
    Jtest = forest.predict(xtest)
    Jtests.append(accuracy_score(Jtest, ytest))
SHOW = False
if SHOW:
    plt.plot(Jtrains, label="Train accuracy")
    plt.plot(Jtests, c="r", label="Test accuracy")
    plt.xticks(ticks=range(len(max_depths)), labels=max_depths)
    plt.xlabel("max tree depth", fontsize=12)
    plt.ylabel("accuracy", fontsize=12)
    plt.title("Accuracy on max tree depth", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", lw=0.5)
    plt.tight_layout()
    plt.show()


### VALIDATE NUM OF TREES ###
n_estimators = [10, 50, 100, 500]
Jtrains = []
Jtests = []
for trees in n_estimators:
    forest = RandomForestClassifier(n_estimators=trees, random_state=RANDOM_STATE)
    forest.fit(xtrain, ytrain)
    Jtrain = forest.predict(xtrain)
    Jtrains.append(accuracy_score(Jtrain, ytrain))
    Jtest = forest.predict(xtest)
    Jtests.append(accuracy_score(Jtest, ytest))
SHOW = False
if SHOW:
    plt.plot(Jtrains, label="Train accuracy")
    plt.plot(Jtests, c="r", label="Test accuracy")
    plt.xticks(ticks=range(len(n_estimators)), labels=n_estimators)
    plt.xlabel("num of trees", fontsize=12)
    plt.ylabel("accuracy", fontsize=12)
    plt.title("Accuracy on num of estimators", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", lw=0.5)
    plt.tight_layout()
    plt.show()


### DECISION MAKING ###
forest = RandomForestClassifier(min_samples_split=10, max_depth=8, n_estimators=100)
forest.fit(xtrain, ytrain)
Jtrain = accuracy_score(forest.predict(xtrain), ytrain)
Jtest = accuracy_score(forest.predict(xtest), ytest)
print(f"score in train (random forest) = {Jtrain:.4f}")
print(f"score in test (random forest) = {Jtest:.4f}")


### GRID SEARCH CV ###
params = {
    "min_samples_split": [2, 10, 30, 50, 100, 200, 300, 700],
    "max_depth": [2, 4, 8, 16, 32, 64, None],
    "n_estimators": [10, 50, 100, 500],
}
forest = RandomForestClassifier()
grid_search = GridSearchCV(
    estimator=forest, param_grid=params, cv=5, scoring="accuracy"
)
# grid_search.fit(xtrain, ytrain)
# best_params = grid_search.best_params_
# best_forest = grid_search.best_estimator_
"""
Best params : {
    'max_depth': 32,
    'min_samples_split': 10,
    'n_estimators': 100
    }
"""
forest.set_params(max_depth=32, min_samples_split=10, n_estimators=100)
forest.fit(xtrain, ytrain)
Jtrain = accuracy_score(forest.predict(xtrain), ytrain)
Jtest = accuracy_score(forest.predict(xtest), ytest)
print(f"score in train (best forest) = {Jtrain:.4f}")
print(f"score in test (best forest) = {Jtest:.4f}")


### XGBOOST ###
### USE VALIDATION SET FOR THE EARLY STOPPING ROUNDS ###
xtrain, xdev, ytrain, ydev = train_test_split(
    xtrain, ytrain, test_size=0.20, random_state=RANDOM_STATE
)
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    gamma=0.1,
    verbosity=1,
    random_state=RANDOM_STATE,
)
xgb.fit(xtrain, ytrain, eval_set=[(xdev, ydev)], early_stopping_rounds=50)
print(f"Best iteration : {xgb.best_iteration}")
Jtrain = accuracy_score(xgb.predict(xtrain), ytrain)
Jtest = accuracy_score(xgb.predict(xtest), ytest)
print(f"score in train (xgboost) = {Jtrain:.4f}")
print(f"score in test (xgboost) = {Jtest:.4f}")


# # ### GRID SEARCH CV ###
# params = {
#     "max_depth": [3, 5, 7, None],
#     "min_child_weight": [1, 3, 5, None],
#     "gamma": [0, 0.1, 0.2, None],
#     "subsample": [0.6, 0.8, 1.0, None],
#     "colsample_bytree": [0.6, 0.8, 1.0, None],
#     "learning_rate": [0.05, 0.1, 0.2],
# }
# xgb = XGBClassifier(n_estimators=500, random_state=RANDOM_STATE)
# GridSearch = GridSearchCV(estimator=xgb, param_grid=params, cv=None, scoring="accuracy", verbose=False)
# GridSearch.fit(xtrain, ytrain, eval_set=[(xdev, ydev)], early_stopping_rounds=30, verbose=1)
# best_params = GridSearch.best_params_
# best_xgb = GridSearch.best_estimator_
# Jtrain = accuracy_score(best_xgb.predict(xtrain), ytrain)
# Jtest = accuracy_score(best_xgb.predict(xtest), ytest)
# print(f"Best parameters: {best_params}")
# print(f"Score in train (XGBoost) = {Jtrain:.4f}")
# print(f"Score in test (XGBoost) = {Jtest:.4f}")