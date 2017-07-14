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


# for i in range(1,14):
#     path = '/cs/home/jf231/Dissertation/CS5099/PCA/Train/' + str(i) + '/'
#     os.makedirs(path)
#     path = '/cs/home/jf231/Dissertation/CS5099/PCA/Test1/' + str(i) + '/'
#     os.makedirs(path)
#     path = '/cs/home/jf231/Dissertation/CS5099/PCA/Test2/' + str(i) + '/'
#     os.makedirs(path)


# path = '/cs/home/jf231/Dissertation/CS5099/All_Images/Set_2/'
# for i in range(1, 14):
#     destination = '/cs/home/jf231/Dissertation/CS5099/PCA/Train/' + str(i) + '/'
#     count = 0
#     for m in os.listdir(path):
#         id = int(m[5:7])
#         if id == i:
#             img = cv2.imread(path + m, 0)
#             cv2.imwrite(destination + m, img)
#             count += 1
#             if count == 2:
#                 break
#
# for i in range(15, 40):
#     destination = '/cs/home/jf231/Dissertation/CS5099/PCA/Train/' + str(i) + '/'
#     count = 0
#     for m in os.listdir(path):
#         id = int(m[5:7])
#         if id == i:
#             img = cv2.imread(path + m, 0)
#             cv2.imwrite(destination + m, img)
#             count += 1
#             if count == 2:
#                 break

########################################################################################################################
# def Gaussain2D( h1, w1, h2, w2, sigma):
#     x0 = int((w1 + w2) / 2)
#     y0 = int((h1 + h2) / 2)
#     x = linspace(w1, w2 - 1, (w2 - w1))
#     y = linspace(h1, h2 - 1, (h2 - h1))
#     x, y = meshgrid(x, y)
#     # para = math.sqrt(2 * math.pi) * sigma
#     gaus = exp(-(((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2))))
#     return gaus
#
# def Gaussain_Border( h1, w1, h2, w2, sigma):
#     # gaus = mat(zeros(((h2 - h1 - 1), (w2 - w1 - 1))))
#     xc = int((w1 + w2) / 2)
#     yc = int((h1 + h2) / 2)
#     x = linspace(w1, xc - 1, (xc - w1))
#     y = linspace(h1, yc - 1, (yc - h1))
#     x, y = meshgrid(x, y)
#     gaus_1 = exp(-(((x - w1) ** 2 + (y - h1) ** 2) / (2 * (sigma ** 2))))
#     x = linspace(xc, w2 - 1, (w2 - xc))
#     y = linspace(h1, yc - 1, (yc - h1))
#     x, y = meshgrid(x, y)
#     gaus_2 = exp(-(((x - w2 + 1) ** 2 + (y - h1) ** 2) / (2 * (sigma ** 2))))
#     x = linspace(w1, xc - 1, (xc - w1))
#     y = linspace(yc, h2 - 1, (h2 - yc))
#     x, y = meshgrid(x, y)
#     gaus_3 = exp(-(((x - w1) ** 2 + (y - h2 + 1) ** 2) / (2 * (sigma ** 2))))
#     x = linspace(xc, w2 - 1, (w2 - xc))
#     y = linspace(yc, h2 - 1, (h2 - yc))
#     x, y = meshgrid(x, y)
#     gaus_4 = exp(-(((x - w2 + 1) ** 2 + (y - h2 + 1) ** 2) / (2 * (sigma ** 2))))
#     t1 = concatenate((gaus_1, gaus_2), axis=1)
#     t2 = concatenate((gaus_3, gaus_4), axis=1)
#     gaus = 1- concatenate((t1, t2))
#     return gaus
#
# # res = Gaussain_Border(0,0,24,21,10)
# res = Gaussain2D(0,0,24,21,6)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# x = linspace(0, 20, (21))
# y = linspace(0, 23, (24))
# x, y = meshgrid(x, y)
# ax.set_xlabel('horizonal')
# ax.set_ylabel('vertical')
# ax.set_zlabel('weight')
# surf = ax.plot_surface(x, y, res, rstride=1, cstride=1, cmap=cm.YlGnBu)
# plt.title('The weight for each pixel',fontsize=10)
# plt.show()
########################################################################################################################
# set_3 = pd.read_csv('set3_Gau_border_select_para.csv')
# set_4 = pd.read_csv('set4_Gau_border_select_para.csv')
# set_3 = np.array(set_3)
# set_4 = np.array(set_4)
# overlap_ratio = np.array([0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
# sigma = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
# labels = ['0.6', '1', '2', '4', '6', '8', '9', '10', '11']
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# x, y = np.meshgrid(sigma, overlap_ratio)
# print(np.shape(x))
# ax.set_xlabel('sigma')
# ax.set_ylabel('overlap ratio')
# ax.set_zlabel('Recognition Rate')
# plt.xticks(sigma, labels)
# # YlOrRd, RdPu
# surf = ax.plot_surface(x, y, set_4, rstride=1, cstride=1, cmap=cm.YlGnBu)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.title('Image Set 4')
# plt.show()
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
# set_3 = pd.read_csv('Original_Shift6_3.csv')
# set_3 = pd.read_csv('Improved_Shift6_3.csv')
# set_3 = pd.read_csv('Gau_central_sigma4_3.csv')
# set_3 = pd.read_csv('Gau_Border_Shift6_3.csv')
# set_3 = pd.read_csv('Gau_Central_Sigma6_3.csv')
# set_3= pd.read_csv('Combination2_Shift6_3.csv')
# set_3 = np.array(set_3)
# print(np.amax(set_3.flatten()))
# print(np.amin(set_3.flatten()))
# print(np.mean(set_3.flatten()))
# print(np.std(set_3.flatten()))
# print(np.cov(set_3.flatten()))
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# shift_vertical = np.array([-6, -4, -2, 0, 2, 4, 6])
# shift_horizon = np.array([-6, -4, -2, 0, 2, 4, 6])
# x, y = np.meshgrid(shift_horizon, shift_vertical)
# ax.set_xlabel('Horizonal shift')
# ax.set_ylabel('Vertical shift')
# ax.set_zlabel('Recognition Rate')
# ax.set_zlim(0.6,1)
# surf = ax.plot_surface(x, y, set_3, rstride=1, cstride=1, cmap=cm.YlGnBu)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.title('Combined Approach', fontsize=10)
# plt.show()
########################################################################################################################
