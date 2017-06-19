import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import  pandas as pd


def feature_Select(x1, x2, y1, y2):
    importance = np.array([0.0 for i in range(np.shape(x1)[1])])
    extra = np.concatenate((x2,y2.reshape(np.shape(y2)[0],1)),axis=1)
    extra = pd.DataFrame(extra)
    for i in range(5):
        print('Round %d:' % i)
        # Remain_X, Select_X, y_remain, y_select = train_test_split(x2, y2, test_size=0.04)
        sample = extra.sample(frac=0.04,replace=False)
        sample = np.array(sample)
        print(np.shape(sample))
        X = np.concatenate((x1, sample[:,0:49]))
        y = np.concatenate((y1, sample[:,49]))
        normalized_X = preprocessing.normalize(X)
        X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.3, random_state=111)
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
    importance = np.around(importance/5,decimals=5)
    print(importance)
    return importance
