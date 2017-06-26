import matplotlib.pyplot as plt
import cv2
import os
import math
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

set_4_exp = np.array(
    [0.817, 0.817, 0.825, 0.825, 0.817, 0.817, 0.817, 0.833, 0.833, 0.833, 0.833, 0.833, 0.833, 0.825, 0.825, 0.825,
     0.808, 0.8])
set_5_exp = np.array(
    [0.383, 0.39, 0.4, 0.411, 0.411, 0.411, 0.417, 0.422, 0.428, 0.439, 0.45, 0.444, 0.45, 0.45, 0.456, 0.45, 0.439,
     0.411])
set_4_log = np.array(
    [0.808, 0.808, 0.817, 0.817, 0.817, 0.817, 0.817, 0.817, 0.817,0.817, 0.817, 0.808, 0.808, 0.808, 0.808, 0.8, 0.8, 0.8])
parameters = np.array([1, 20, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 180, 200, 250, 300])
plt.figure()
plt.plot(parameters, set_4_log, c='c', ls='-.', label='Logarithm', linewidth=3)
plt.plot(parameters, set_4_exp, c='#ff6347', ls='-.', label='Exponentiation', linewidth=3)
plt.ylim((0.79,0.85))
plt.xlabel('parameter')
plt.ylabel('Mean Recognition Rate')
plt.title('Recognition rates on set 4')
plt.legend()
plt.show()

# plt.plot(parameters, set_5_exp, c='#ff6347', ls='-.', label='Exponentiation', linewidth=3)
# plt.ylim((0.38,0.47))
# plt.xlabel('parameter')
# plt.ylabel('Mean Recognition Rate')
# plt.title('Recognition rates on set 5')
# plt.legend()
# plt.show()


# path = '/cs/home/jf231/Dissertation/CS5099/PartImages/'
# destinations = ['/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_1/',
#                 '/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_2/',
#                 '/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_3/',
#                 '/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_4/',
#                 '/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_5/',
#                 '/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_6/']
# for m in os.listdir(path):
#     if (len(m) == 24):
#         horizon = int(m[13:16])
#         vertical = int(m[18:20])
#         angle = int((horizon + vertical) / 2)
#         img = cv2.imread(path + m, 0)
#         if angle < 12:
#
#             cv2.imwrite(destinations[0] + m, img)
#         elif angle < 22:
#
#             cv2.imwrite(destinations[1] + m, img)
#         elif angle < 37:
#
#             cv2.imwrite(destinations[2] + m, img)
#         elif angle < 50 and vertical < 90 and horizon < 90:
#
#             cv2.imwrite(destinations[3] + m, img)
#         elif angle < 75 and vertical < 90 and horizon < 110:
#
#             cv2.imwrite(destinations[4] + m, img)
#         else:
#             cv2.imwrite(destinations[5] + m, img)

# for m in os.listdir(path):
# if (len(m) == 24):
#         horizon = int(m[13:16])
#         vertical = int(m[18:20])
#         OB = 1.0
#         BC = math.tan(math.pi / 180 * horizon) * OB
#         AB = math.tan(math.pi / 180 * vertical) * OB
#         BD = math.sqrt(BC ** 2 + AB ** 2) / 2
#         angle = int(math.atan(BD / OB) * (180 / math.pi))
#         img = cv2.imread(path + m, 0)
#         if horizon < 90 and vertical < 90:
#             if angle < 11:
#                 cv2.imwrite(destinations[0] + m, img)
#             elif angle < 20:
#                 cv2.imwrite(destinations[1] + m, img)
#             elif angle < 40:
#                 cv2.imwrite(destinations[2] + m, img)
#             elif angle < 80:
#                 cv2.imwrite(destinations[3] + m, img)
#             else:
#                 cv2.imwrite(destinations[4] + m, img)
#         else:
#             cv2.imwrite(destinations[4] + m, img)


# str = 'yaleB01_P00A-010E+00.pgm'
# print(int(str[5:7]))
# print(int(str[13:16]))
# print(int(str[18:20]))
# str = '0005'
# print(int(str))
# region_type = np.dtype({
#     'names': ['region_height', 'region_width'],
#     'formats': ['i', 'i']
# })
# sub_regions = np.array([(24, 21), (24, 24), (27, 24)], dtype=region_type)


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
