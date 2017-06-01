
from LBP_Implement import LBP_Implement
from numpy import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    R = 1
    P = 8
    type = 0  # 1 for original, 0 for circular neighbor-sets
    uniform = 1  # 1: use uniform patterns, 0: not
    win_num = 7
    obj = LBP_Implement(R, P, type, uniform, win_num)
    path = 'F:/dissertation/PartImages/'
    character = 'P00A+000E+00.pgm'
    obj.run_LBP(path, character)
    characters = array(['P00A+000E+45.pgm', 'P00A+020E-10.pgm', 'P00A-060E-20.pgm'])
    for char in characters:
        acc = obj.calculate_Accuracy(path, char)
        print(acc)
    ####################################################################################################################
    # Choose Radius, Number of sampling points, number of local regions
    # Choose 2, 8, 7
    # rate = array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # # win_size = array(['42*48', '33*38', '28*32', '24*27', '21*24', '18*21', '16*19'])
    # win_num = array([4, 5, 6, 7, 8, 9, 10])
    # characters = array(['P00A+000E+45.pgm', 'P00A+000E+20.pgm'])
    # index = 0
    # for j in range(7):
    #     LBP_V1.run_LBP(path, character, type, uniform, 1, 8, win_num[j])
    #     temp = 0.0
    #     for c in characters:
    #         temp += LBP_V1.calculate_Accuracy(path, c)
    #     rate[0, index] = temp / 2
    #     index = index + 1
    # print(rate[0])
    # index = 0
    # for j in range(7):
    #     LBP_V1.run_LBP(path, character, type, uniform, 2, 8, win_num[j])
    #     temp = 0.0
    #     for c in characters:
    #         temp += LBP_V1.calculate_Accuracy(path, c)
    #     rate[1, index] = temp / 2
    #     index = index + 1
    # print(rate[1])
    # index = 0
    # for j in range(7):
    #     LBP_V1.run_LBP(path, character, type, uniform, 2, 16, win_num[j])
    #     temp = 0.0
    #     for c in characters:
    #         temp += LBP_V1.calculate_Accuracy(path, c)
    #     rate[2, index] = temp / 2
    #     index = index + 1
    #
    # print(rate[2])
    # plt.figure()
    # plt.plot(win_num, rate[0], c='c', ls='-.', label='LBP 8,1')
    # plt.plot(win_num, rate[1], c='#ffa500', ls='-.', label='LBP 8,2')
    # plt.plot(win_num, rate[2], c='#ff6347', ls='-.', label='LBP 16,2')
    # plt.legend()
    # plt.xlabel('Window Size')
    # plt.ylabel('Mean Recognition Rate')
    # plt.show()
    ####################################################################################################################
