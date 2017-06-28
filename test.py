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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

scale_ratio = np.array([0.9, 0.915, 0.925, 0.935, 0.95, 0.965, 0.975, 0.985, 1, 1.015, 1.025, 1.035, 1.05, 1.065, 1.075,
                        1.085, 1.1])
improved_4 = np.array(
    [0.733, 0.725, 0.8, 0.808, 0.883, 0.9, 0.875, 0.892, 0.942, 0.858, 0.875, 0.85, 0.842, 0.8, 0.775, 0.775, 0.725])
original_4 = np.array(
    [0.55, 0.617, 0.55, 0.6, 0.692, 0.7, 0.65, 0.667, 0.708, 0.575, 0.6, 0.592, 0.55, 0.467, 0.475, 0.475, 0.467])

improved_3 = np.array(
    [0.891, 0.903, 0.915, 0.915, 0.952, 0.952, 0.939, 0.939, 0.958, 0.939, 0.933, 0.927, 0.903, 0.921, 0.915, 0.915,
     0.842])
original_3 = np.array(
    [0.794, 0.818, 0.812, 0.824, 0.824, 0.830, 0.824, 0.824, 0.830, 0.818, 0.824, 0.818, 0.788, 0.788, 0.764, 0.764,
     0.703])
print(np.cov(improved_3))
print(np.cov(original_3))
plt.figure()
# plt.plot(scale_ratio, improved_4)
# plt.plot(scale_ratio, original_4)
coe = np.polyfit(scale_ratio, improved_3, 4)
poly = np.poly1d(coe)
xs = np.arange(0.9, 1.1, 0.01)
ys = poly(xs)
plt.plot(scale_ratio, improved_3, 'o')
plt.plot(xs, ys)

coe = np.polyfit(scale_ratio, original_3, 4)
poly = np.poly1d(coe)
xs = np.arange(0.9, 1.1, 0.01)
ys = poly(xs)
plt.plot(scale_ratio, original_3, 'o')
plt.plot(xs, ys)
plt.ylim(0.7, 1)
plt.show()
########################################################################################################################
# Draw 3d graphs for shifting, on image set 3
# import pandas as pd
# improved_rates = pd.read_csv('improved_rates.csv')
# original_rates = pd.read_csv('original_rates.csv')
# improved_rates = np.array(improved_rates)
# original_rates = np.array(original_rates)
# print(np.cov(improved_rates.flatten()))
# print(np.cov(original_rates.flatten()))
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# shift_x = np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8])
# shift_y = np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8])
# x,y = np.meshgrid(shift_x,shift_y)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('Recognition Rate')
# ax.scatter(x, y, improved_rates, marker='o',color = 'c')
# ax.scatter(x, y, original_rates, marker='^',color = '#ff6347')
# ax.plot_wireframe(x, y, improved_rates, color='c', linewidth=2, label='Improved Rates')
# ax.plot_wireframe(x, y, original_rates, color='#ff6347', linewidth = 2,label='Original Rates')
# plt.legend()
# plt.show()
########################################################################################################################
# Select suitable parameter for exponentiation and logarithm
# set_4_exp = np.array(
#     [0.817, 0.817, 0.825, 0.825, 0.817, 0.817, 0.817, 0.833, 0.833, 0.833, 0.833, 0.833, 0.833, 0.825, 0.825, 0.825,
#      0.808, 0.8])
# set_5_exp = np.array(
#     [0.383, 0.39, 0.4, 0.411, 0.411, 0.411, 0.417, 0.422, 0.428, 0.439, 0.45, 0.444, 0.45, 0.45, 0.456, 0.45, 0.439,
#      0.411])
# set_4_log = np.array(
#     [0.808, 0.808, 0.817, 0.817, 0.817, 0.817, 0.817, 0.817, 0.817,0.817, 0.817, 0.808, 0.808, 0.808, 0.808, 0.8, 0.8, 0.8])
# parameters = np.array([1, 20, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 180, 200, 250, 300])
# plt.figure()
# plt.plot(parameters, set_4_log, c='c', ls='-.', label='Logarithm', linewidth=3)
# plt.plot(parameters, set_4_exp, c='#ff6347', ls='-.', label='Exponentiation', linewidth=3)
# plt.ylim((0.79,0.85))
# plt.xlabel('parameter')
# plt.ylabel('Mean Recognition Rate')
# plt.title('Recognition rates on set 4')
# plt.legend()
# plt.show()

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
####################################################################################################################
