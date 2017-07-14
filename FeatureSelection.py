import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def feature_Select(x1, x2, y1, y2, regions):
    importance = np.array([0.0 for i in range(np.shape(x1)[1])])
    extra = np.concatenate((x2, y2.reshape(np.shape(y2)[0], 1)), axis=1)
    extra = pd.DataFrame(extra)
    for i in range(5):
        print('Round %d:' % i)
        # Remain_X, Select_X, y_remain, y_select = train_test_split(x2, y2, test_size=0.04)
        sample = extra.sample(frac=0.2, replace=False)
        sample = np.array(sample)
        # print(np.shape(sample))
        X = np.concatenate((x1, sample[:, 0:regions]))
        y = np.concatenate((y1, sample[:, regions]))
        normalized_X = preprocessing.normalize(X)
        X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.25, random_state=111)
        print('Adaboost')
        # abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        # parameters = {'n_estimators': (150,160,170),
        #               'base_estimator__max_depth': (2,3,4),
        #               'learning_rate': (0.6, 0.8, 1.0),
        #               'algorithm': ('SAMME', 'SAMME.R')}
        # clf = GridSearchCV(abc, parameters)
        # clf.fit(X_train,y_train)
        # print('best parameter:')
        # print(clf.best_params_)

        abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
                                 n_estimators=180,
                                 learning_rate=0.8,
                                 algorithm='SAMME.R')
        abc.fit(X_train, y_train)
        importance += np.around(abc.feature_importances_, decimals=5)
        y_pred = abc.predict(X_test)
        print(confusion_matrix(y_test, y_pred))

        # print('Random forest:')
        # rf = RandomForestClassifier()
        # parameters = {'n_estimators': (70,80,90,100,110)}
        # clf = GridSearchCV(rf, parameters)
        # clf.fit(X_train, y_train)
        # print('best parameter:')
        # print(clf.best_params_)

        # rf = RandomForestClassifier(n_estimators=90)
        # rf.fit(X_train, y_train)
        # importance += np.around(rf.feature_importances_, decimals=4)
        # y_pred = rf.predict(X_test)
        # print(confusion_matrix(y_test, y_pred))
    importance = np.around(importance / 5, decimals=5)
    temp = importance.reshape(int(np.sqrt(regions)), int(np.sqrt(regions)))
    H, W = np.shape(temp)
    adjust = int(W / 2)
    for row in range(H):
        for col in range(adjust):
            average = (temp[row][col] + temp[row][W - col - 1]) / 2
            temp[row][col] = average
            temp[row][W - col - 1] = average
    temp = temp.flatten()
    sorted_weights = sorted(temp)
    thresholds = [0.1, 0.8, 0.9, 1]
    weight_standard = [0, 1, 2, 4]
    start = -1
    for t in range(4):
        index = int(regions * thresholds[t]) - 1
        end = sorted_weights[index]
        for j in range(regions):
            if ((temp[j] <= end) and (temp[j] > start)):
                temp[j] = weight_standard[t]
        start = end
    importance = temp
    # print(importance)
    return importance
