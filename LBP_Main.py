from LBP_Implement import LBP_Implement
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from FeatureSelection import feature_Select
import cv2
import multiprocessing as mp
from PCA_SVD import PCA_SVD

if __name__ == '__main__':
    R = 1
    P = 8
    type = 1  # 1 for original, 0 for circular neighbor-sets
    uniform = 1  # 1: use uniform patterns, 0: not
    w_num = 8
    h_num = 8
    isgradiented = 1
    exp_para = 120
    sigma = 6
    gap_size = 0
    shift_vertical = 0
    shift_horizon = 0
    overlap_ratio = 0.3
    # paths = ['F:/dissertation/Yale_images/Set_1/',
    #          'F:/dissertation/Yale_images/Set_2/',
    #          'F:/dissertation/Yale_images/Set_3/',
    #          'F:/dissertation/Yale_images/Set_4/',
    #          'F:/dissertation/Yale_images/Set_5/',
    #          'F:/dissertation/Yale_images/Set_6/']
    paths = ['/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_1/',
             '/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_2/',
             '/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_3/',
             '/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_4/',
             '/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_5/',
             '/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_6/']
    # pool = mp.Pool(processes=3)
    # shift_vertical = np.array([-6, -4, -2, 0, 2, 4, 6])
    # shift_horizon = np.array([-6, -4, -2, 0, 2, 4, 6])
    # set_3 = np.mat(np.zeros((len(shift_vertical), len(shift_horizon))))
    # obj = LBP_Implement(R, P, type, uniform, w_num, h_num, overlap_ratio, gap_size)
    # obj.run_LBP(paths[0], isgradiented, exp_para, sigma)
    # for i in range(len(shift_vertical)):
    #     for j in range(len(shift_horizon)):
    #         set_3[i,j]=obj.calculate_Accuracy(paths[2],shift_vertical[i],shift_horizon[j])
    #     print(set_3[i,:])
    weights = []
    obj = LBP_Implement(R, P, type, uniform, w_num, h_num, overlap_ratio, gap_size,weights)
    obj.run_LBP(paths[0], isgradiented, exp_para, sigma)
    for i in range(2,6):
        acc = obj.calculate_Accuracy(paths[i],0,0)
        print('Accuracy: %10.3f'%acc)

    # train_path = '/cs/home/jf231/Dissertation/CS5099/PCA_1/Train/'
    # test_paths = ['/cs/home/jf231/Dissertation/CS5099/PCA_1/Test1/',
    #               '/cs/home/jf231/Dissertation/CS5099/PCA_1/Test2/']
    # pool = mp.Pool(processes=2)
    # obj = LBP_Implement(R, P, type, uniform, w_num, h_num, overlap_ratio, gap_size)
    # print('Training...')
    # obj.run_LBP(train_path, isgradiented, exp_para, sigma)
    # obj.calculate_Weights(paths[1])
    # print('Testing...')
    # acc = pool.map(obj.calculate_Accuracy,test_paths)
    # print(acc)

