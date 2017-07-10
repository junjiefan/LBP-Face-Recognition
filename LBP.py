from LBP_Implement import LBP_Implement
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from FeatureSelection import feature_Select
import cv2
import multiprocessing as mp
from LBP_PCA_SVD import LBP_PCA

if __name__ == '__main__':
    R = 1
    P = 8
    type = 1  # 1 for original, 0 for circular neighbor-sets
    uniform = 1  # 1: use uniform patterns, 0: not
    w_num = 8
    h_num = 8
    isgradiented = 0
    exp_para = 0
    sigma = 10
    gap_size = 0
    shift_vertical = 0
    shift_horizon = 0
    overlap_ratio = 0
    paths = ['F:/dissertation/Yale_images/Set_1/',
             'F:/dissertation/Yale_images/Set_2/',
             'F:/dissertation/Yale_images/Set_3/',
             'F:/dissertation/Yale_images/Set_4/',
             'F:/dissertation/Yale_images/Set_5/',
             'F:/dissertation/Yale_images/Set_6/']
    # paths = ['/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_1/',
    #          '/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_2/',
    #          '/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_3/',
    #          '/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_4/',
    #          '/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_5/',
    #          '/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_6/']
    # pool = mp.Pool(processes=3)
    # shift_vertical = np.array([-6, -4, -2, 0, 2, 4, 6])
    # shift_horizon = np.array([-6, -4, -2, 0, 2, 4, 6])
    # set_3 = np.mat(np.zeros((len(shift_vertical), len(shift_horizon))))
    #
    # obj = LBP_Implement(R, P, type, uniform, w_num, h_num, overlap_ratio, gap_size)
    # obj.run_LBP(paths[0], isgradiented, exp_para, sigma)
    # for i in range(len(shift_vertical)):
    #     for j in range(len(shift_horizon)):
    #         set_3[i,j]=obj.calculate_Accuracy(paths[2],shift_vertical[i],shift_horizon[j])
    #     print(set_3[i,:])

    # obj = LBP_Implement(R, P, type, uniform, w_num, h_num, overlap_ratio, gap_size)
    # obj.run_LBP(paths[0], isgradiented, exp_para, sigma)
    # for i in range(2,6):
    #     acc = obj.calculate_Accuracy(paths[i],0,0)
    #     print('Accuracy: %10.3f'%acc)
    path = 'F:/dissertation/Test/'
    lp = LBP_PCA(R, P, type, uniform, w_num, h_num, overlap_ratio, gap_size)
    lp.run_LBP(path, isgradiented, exp_para, sigma)
    # acc = lp.calculate_Accuracy(paths[0], 0, 0)
    # print('Accuracy: %10.3f' % acc)
