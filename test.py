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
import pandas as pd

########################################################################################################################
# set_3 = pd.read_csv('set3_Gau_para.csv')
# set_4 = pd.read_csv('set4_Gau_para.csv')
# set_3 = np.array(set_3)
# set_4 = np.array(set_4)
# overlap_ratio = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
# # sigma = np.array([0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7])
# sigma = np.array([0, 1, 2, 3, 4, 5, 6, 7])
# labels = ['0.4', '0.6', '0.8', '1', '2', '3', '4', '5']
# set_4 = set_4[:, 0:8]
# # sigma = np.array([1, 2, 3, 4, 5, 6, 7])
# # labels = ['0.6', '0.8', '1', '2', '3', '4', '5']
# # set_3 = set_3[:, 1:8]
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# x, y = np.meshgrid(sigma, overlap_ratio)
# print(np.shape(x))
# ax.set_xlabel('sigma')
# ax.set_ylabel('overlap ratio')
# ax.set_zlabel('Recognition Rate')
# plt.xticks(sigma, labels)
# #YlOrRd, RdPu
# surf = ax.plot_surface(x,y,set_4,rstride=1, cstride=1, cmap=cm.YlGnBu)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()
########################################################################################################################
# Draw 3d graphs for shifting, on image set 3
# improved_rates = pd.read_csv('set3_improved_rates.csv')
# original_rates = pd.read_csv('set3_original_rates.csv')
# improved_rates = np.array(improved_rates)
# original_rates = np.array(original_rates)
# print('Variance:')
# print(np.cov(improved_rates.flatten()))
# print(np.cov(original_rates.flatten()))
# print('Standard deviation: ')
# print(np.std(improved_rates.flatten()))
# print(np.std(original_rates.flatten()))
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# shift_vertical = np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8])
# shift_horizon = np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8])
# x, y = np.meshgrid(shift_horizon, shift_vertical)
# ax.set_xlabel('horizonal shift')
# ax.set_ylabel('vertical shift')
# ax.set_zlabel('Recognition Rate')
# surf = ax.plot_surface(x, y, improved_rates, rstride=1, cstride=1, cmap=cm.Purples)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# # surf = ax.plot_surface(x, y, original_rates, rstride=1, cstride=1, cmap=cm.RdPu)
# # fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.title('Image gradient', fontsize = 10)
# plt.show()
########################################################################################################################
# 3 using Gaussian
# set_3_Gau = pd.read_csv('set3_Gau_shift.csv')
# set_3_Gau = np.array(set_3_Gau)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# shift_vertical = np.array([-6, -4, -2, 0, 2, 4, 6])
# shift_horizon = np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8])
# x, y = np.meshgrid(shift_horizon, shift_vertical)
# ax.set_xlabel('Horizonal shift')
# ax.set_ylabel('Vertical shift')
# ax.set_zlabel('Recognition Rate')
# ax.set_zlim(0.1, 1)
# surf = ax.plot_surface(x, y, set_3_Gau, rstride=1, cstride=1, cmap=cm.YlGnBu)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.title('Gaussian founction', fontsize=10)
# plt.show()
########################################################################################################################
