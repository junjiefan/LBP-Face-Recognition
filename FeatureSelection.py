import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def feature_Select(x1, x2, y1, y2):
    # len1 = len(y1)
    # extra = np.concatenate((x2, y2.reshape(y2.shape[0], 1)), axis=1)
    # sample = np.array(np.random.sample(list(extra), len1 * 3))
    Remain_X, Select_X, y_remain, y_select = train_test_split(x2, y2, test_size=0.05, random_state=111)
    print(np.shape(Select_X))

    X = np.concatenate((x1, Select_X))
    y = np.concatenate((y1, y_select))
    normalized_X = preprocessing.normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.3, random_state=111)
    # weights = np.array([1.0 / X.shape[0]
    #                    for i in range(X.shape[0])])
    print('Adaboost')
    # abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
    # parameters = {'n_estimators': (140,160,180,200),
    #               'base_estimator__max_depth': (2,3,4,5),
    #               'learning_rate': (0.6, 0.8, 1.0),
    #               'algorithm': ('SAMME', 'SAMME.R')}
    # clf = GridSearchCV(abc, parameters)
    # clf.fit(X_train,y_train)
    # print('best parameter:')
    # print(clf.best_params_)

    abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
                             n_estimators= 160,
                             learning_rate=0.8,
                             algorithm='SAMME.R')
    abc.fit(X_train, y_train)
    print('Feature importance:')
    abc_importance = np.around(abc.feature_importances_,decimals=5)
    print(abc_importance)
    y_pred = abc.predict(X_test)
    print(confusion_matrix(y_test, y_pred))




    print('Random forest:')
    # rf = RandomForestClassifier()
    # parameters = {'n_estimators': (70,80,90,100,110)}
    # clf = GridSearchCV(rf, parameters)
    # clf.fit(X_train, y_train)
    # print('best parameter:')
    # print(clf.best_params_)

    rf = RandomForestClassifier(n_estimators=90)
    rf.fit(X_train, y_train)
    importance_level = np.around(rf.feature_importances_,decimals=4)
    print(importance_level)
    y_pred = rf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    return abc_importance
