from LBP_Implement import LBP_Implement
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from FeatureSelection import feature_Select
import cv2
import multiprocessing as mp

# def test(obj, train_path, test_path):
#     obj.run_LBP(train_path, isgradiented, exp_para, sigma)
#     results = np.array([0.0] * len(test_path))
#     for i in range(len(test_path)):
#         results[i] = obj.calculate_Accuracy(test_path[i], weights)
#     return results


if __name__ == '__main__':
    R = 1
    P = 8
    type = 1  # 1 for original, 0 for circular neighbor-sets
    uniform = 1  # 1: use uniform patterns, 0: not
    w_num = 8
    h_num = 8
    isgradiented = 0
    exp_para = 0
    sigma = 2
    gap_size = 6
    shift_vertical = 6
    shift_horizon =6
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

    weights = []
    obj = LBP_Implement(R, P, type, uniform, w_num, h_num, overlap_ratio, gap_size)
    obj.run_LBP(paths[0], isgradiented, exp_para, sigma)
    for i in range(2,6):
            acc = obj.calculate_Accuracy(paths[i], weights, shift_vertical, shift_horizon)
            print('Accuracy: %10.3f'%acc)


