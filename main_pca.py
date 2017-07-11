from PCA_SVD import PCA_SVD
from LBP_Implement import LBP_Implement
import numpy as np

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
    ################################
    train_path = [''] * 10
    train_ids = np.array([0] * 10)
    for i in range(1, 11):
        train_path[i - 1] = 'F:/dissertation/PCA/Train/' + str(i) + '/'
        train_ids[i - 1] = i

    test_path = [''] * 10
    test_ids = np.array([0] * 10)
    for i in range(1, 11):
        test_path[i - 1] = 'F:/dissertation/PCA/Test/' + str(i) + '/'
        test_ids[i - 1] = i
    train_matrix = np.mat(np.zeros((10, 37760)))
    test_matrix = np.mat(np.zeros((10, 37760)))
    lp = LBP_Implement(R, P, type, uniform, w_num, h_num, overlap_ratio, gap_size)
    ps = PCA_SVD(10)
    for i in range(10):
        train_matrix[i,:] = lp.run_LBP(train_path[i], isgradiented, exp_para, sigma)
        test_matrix[i,:] = lp.run_LBP(test_path[i], isgradiented, exp_para, sigma)
    # b1 = ps.pca(train_matrix[1,:])
    # b2 = ps.pca(test_matrix[1,:])
    # m = (b1.reshape(3776,5).T)*(b2.reshape(3776,5))
    # sigma = ps.svd(m)
    # print(sigma)
    ps.run_PCA(train_matrix,train_ids,test_matrix,test_ids)



