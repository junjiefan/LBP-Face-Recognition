import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def feature_Select(X,y):
    print(np.shape(X))
    print(np.shape(y))
    print(X[0:5,0:5])
    print(y[0:5])
    # weights = np.array([1.0 / X.shape[0]
    #                    for i in range(X.shape[0])])
    abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20),
                         algorithm="SAMME",
                         n_estimators=200)

    abc.fit(X, y)
    print('Feature importance:')
    print(abc.feature_importances_)

