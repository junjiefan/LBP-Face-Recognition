import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

region_type = np.dtype({
    'names': ['region_height', 'region_width'],
    'formats': ['i', 'i']
})
sub_regions = np.array([(24, 21), (24, 24), (27, 24)], dtype=region_type)


# boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
# parameters = {'n_estimators': (10, 12, 15, 17, 20),
#               'base_estimator__max_depth': (1, 2, 3),
#               'learning_rate': (0.6, 0.7, 0.8, 0.9),
#               'algorithm': ('SAMME', 'SAMME.R')}
# clf = GridSearchCV(boost, parameters)
# clf.fit(iris.data, iris.target)
# print(clf.best_params_)
# print(clf.best_estimator_)
#
# boost = RandomForestClassifier()
# parameters = {'n_estimators': (10, 12, 15, 17, 20)}
# clf = GridSearchCV(boost, parameters)
# clf.fit(iris.data, iris.target)
# print(clf.best_params_)
# print(clf.best_estimator_)

####################################################################################################################
# Choose the overlapping size
# characters = array(['P00A+000E-20.pgm', 'P00A+000E-35.pgm', 'P00A+000E+20.pgm', 'P00A+000E+45.pgm'])
# overlap_ratios = array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4])
# acc = [0.0] * 8
# count = 0
# for overlap_ratio in overlap_ratios:
#     obj = LBP_Implement(R, P, type, uniform, w_num, h_num, overlap_ratio)
#     obj.run_LBP(path, character)
#     temp = 0.0
#     for char in characters:
#         temp += obj.calculate_Accuracy(path, char)
#     acc[count] = temp / 4
#     count +=1
# plt.figure()
# plt.plot(overlap_ratios, acc, c='c', ls='-.', linewidth=3)
# plt.xlabel('Overlap Size')
# plt.ylabel('Mean Recognition Rate')
# plt.show()
####################################################################################################################
# Choose Radius, Number of sampling points, number of local regions
# Choose 2, 8, 7
# rate = [([0.0] * 8) for i in range(3)]
# win_num = array([3, 4, 5, 6, 7, 8, 9, 10])
# characters = array(['P00A+000E-20.pgm', 'P00A+000E-35.pgm', 'P00A+000E+20.pgm', 'P00A+000E+45.pgm',])
#
# for j in range(8):
#     obj = LBP_Implement(1, 8, type, uniform, win_num[j], win_num[j],0)
#     obj.run_LBP(path, character)
#     temp = 0.0
#     for c in characters:
#         temp += obj.calculate_Accuracy(path, c)
#     rate[0][j] = temp / 4
# print(rate[0])
#
# for j in range(8):
#     obj = LBP_Implement(2, 8, type, uniform, win_num[j], win_num[j],0)
#     obj.run_LBP(path, character)
#     temp = 0.0
#     for c in characters:
#         temp += obj.calculate_Accuracy(path, c)
#     rate[1][j] = temp / 4
# print(rate[1])
#
# for j in range(8):
#     obj = LBP_Implement(2, 16, type, uniform, win_num[j], win_num[j],0)
#     obj.run_LBP(path, character)
#     temp = 0.0
#     for c in characters:
#         temp += obj.calculate_Accuracy(path, c)
#     rate[2][j] = temp / 4
# print(rate[2])
# plt.figure()
# plt.plot(win_num, rate[0], c='c', ls='-.', label='LBP 8,1',linewidth = 2)
# plt.plot(win_num, rate[1], c='#ffa500', ls='-.', label='LBP 8,2',linewidth = 2)
# plt.plot(win_num, rate[2], c='#ff6347', ls='-.', label='LBP 16,2',linewidth = 2)
# plt.legend(loc=4)
# plt.xlabel('Window Number')
# plt.ylabel('Mean Recognition Rate')
# plt.show()
####################################################################################################################
