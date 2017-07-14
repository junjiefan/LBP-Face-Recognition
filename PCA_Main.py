from PCA_SVD import PCA_SVD
from LBP_Implement import LBP_Implement
import numpy as np
import os

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
    overlap_ratio = 0
    ################################
    img_num = 10
    train_set = 38
    test_set = 38
    weights = []
    # weights= np.genfromtxt('/Set_Matching/weights2.txt',delimiter='.')
    lp = LBP_Implement(R, P, type, uniform, w_num, h_num, overlap_ratio, gap_size, weights)
    dimension = lp.dimension
    ps = PCA_SVD(img_num, dimension)
    train_matrix = np.mat(np.zeros((train_set, dimension * img_num)))
    print('Training...')
    train_path = '/cs/home/jf231/Dissertation/CS5099/PCA/Train/'
    train_ids = np.array([0] * train_set)
    index = 0
    for m in os.listdir(train_path):
        path = train_path + m + '/'
        train_matrix[index, :], train_ids[index] = lp.run_LBP(path, isgradiented, exp_para, sigma)
        index += 1
    ps.trainBases(train_matrix, train_ids)
    print(train_ids)

    print(('Testing...'))
    test_paths = ['/cs/home/jf231/Dissertation/CS5099/PCA/Test1/',
                  '/cs/home/jf231/Dissertation/CS5099/PCA/Test2/']
    set = 0
    for test_path in test_paths:
        test_matrix = np.mat(np.zeros((test_set, dimension * img_num)))
        test_ids = np.array([0] * test_set)
        index = 0
        for m in os.listdir(test_path):
            path = test_path + m + '/'
            test_matrix[index, :], test_ids[index] = lp.run_LBP(path, isgradiented, exp_para, sigma)
            index += 1
        print(test_path)
        print(test_ids)
        acc = ps.getAccuracy(test_matrix, test_ids, set)
        print('Accuracy :%10.3f' % acc)
        set += 1

