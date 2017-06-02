from LBP_Implement import LBP_Implement
from numpy import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    R = 2
    P = 8
    type = 0  # 1 for original, 0 for circular neighbor-sets
    uniform = 1  # 1: use uniform patterns, 0: not
    w_num = 7
    h_num = 7
    #overlap_size = 4
    path = '/cs/home/jf231/Dissertation/CS5099/PartImages/'
    character = 'P00A+000E+00.pgm'
    # obj = LBP_Implement(R, P, type, uniform, w_num, h_num, overlap_size)
    # obj.run_LBP(path, character)
    # character = 'P00A+000E+45.pgm'
    # acc = obj.calculate_Accuracy(path, character)
    # print(acc)
    characters = array(['P00A+000E-20.pgm', 'P00A+000E-35.pgm', 'P00A+000E+20.pgm', 'P00A+000E+45.pgm'])
    overlap_sizes = array([0,1,2,3,4,5,6,7,8,9,10])
    acc = [0.0] * 11
    for overlap_size in overlap_sizes:
        obj = LBP_Implement(R, P, type, uniform, w_num, h_num, overlap_size)
        obj.run_LBP(path, character)
        temp = 0.0
        for char in characters:
            temp += obj.calculate_Accuracy(path, char)
        acc[overlap_size] = temp / 4

    plt.figure()
    plt.plot(overlap_sizes, acc, c='c', ls='-.', linewidth=3)
    plt.xlabel('Overlap Size')
    plt.ylabel('Mean Recognition Rate')
    plt.show()
    # 7*7 resions 0.975
    # 7*8 regions 0.875
    # 8*8 regions 0.9
    ####################################################################################################################
    # Choose Radius, Number of sampling points, number of local regions
    # Choose 2, 8, 7
    # rate = [([0.0] * 7) for i in range(3)]
    # # win_size = array(['42*48', '33*38', '28*32', '24*27', '21*24', '18*21', '16*19'])
    # win_num = array([4, 5, 6, 7, 8, 9, 10])
    # characters = array(['P00A+000E-20.pgm', 'P00A+000E-35.pgm', 'P00A+000E+20.pgm', 'P00A+000E+45.pgm',])
    # for j in range(7):
    #     obj = LBP_Implement(1, 8, type, uniform, win_num[j])
    #     obj.run_LBP(path, character)
    #     temp = 0.0
    #     for c in characters:
    #         temp += obj.calculate_Accuracy(path, c)
    #     rate[0][j] = temp / 4
    # print(rate[0])
    #
    # for j in range(7):
    #     obj = LBP_Implement(2, 8, type, uniform, win_num[j])
    #     obj.run_LBP(path, character)
    #     temp = 0.0
    #     for c in characters:
    #         temp += obj.calculate_Accuracy(path, c)
    #     rate[1][j] = temp / 4
    # print(rate[1])
    #
    # for j in range(7):
    #     obj = LBP_Implement(2, 16, type, uniform, win_num[j])
    #     obj.run_LBP(path, character)
    #     temp = 0.0
    #     for c in characters:
    #         temp += obj.calculate_Accuracy(path, c)
    #     rate[2][j] = temp / 4
    # print(rate[2])
    #
    # plt.figure()
    # plt.plot(win_num, rate[0], c='c', ls='-.', label='LBP 8,1',linewidth = 2)
    # plt.plot(win_num, rate[1], c='#ffa500', ls='-.', label='LBP 8,2',linewidth = 2)
    # plt.plot(win_num, rate[2], c='#ff6347', ls='-.', label='LBP 16,2',linewidth = 2)
    # plt.legend(loc=4)
    # plt.xlabel('Window Number')
    # plt.ylabel('Mean Recognition Rate')
    # plt.show()
    ####################################################################################################################
