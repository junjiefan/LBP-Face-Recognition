import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def feature_Select(x1, x2, y1, y2):
    # len1 = len(y1)
    # extra = np.concatenate((x2, y2.reshape(y2.shape[0], 1)), axis=1)
    # sample = np.array(np.random.sample(list(extra), len1 * 3))
    Remain_X, Select_X, y_remain, y_select = train_test_split(x2, y2, test_size=0.1, random_state=111)
    print(np.shape(Select_X))

    X = np.concatenate((x1, Select_X))
    y = np.concatenate((y1, y_select))
    normalized_X = preprocessing.normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.3, random_state=111)
    # weights = np.array([1.0 / X.shape[0]
    #                    for i in range(X.shape[0])])
    print('Adaboost')
    abc = AdaBoostClassifier(DecisionTreeClassifier(),
                             algorithm="SAMME.R",
                             n_estimators=200)

    abc.fit(X_train, y_train)
    print('Feature importance:')
    print(abc.feature_importances_)
    y_pred = abc.predict(X_test)
    print(np.shape(y_pred))
    print(confusion_matrix(y_test, y_pred))

    # model = ExtraTreesClassifier()
    # model.fit(normalized_X, y)
    # print('Feature Importance: ')
    # print(model.feature_importances_)

    print('Random forest:')
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    print(rf.feature_importances_)
    y_pred = rf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
