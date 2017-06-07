from LBP_Implement import LBP_Implement
from numpy import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    R = 1
    P = 8
    type = 1  # 1 for original, 0 for circular neighbor-sets
    uniform = 1  # 1: use uniform patterns, 0: not
    w_num = 7
    h_num = 7
    overlap_ratio = 0  # from 0 to 1
    path = '/cs/home/jf231/Dissertation/CS5099/Images/'
    # path = 'F:/dissertation/PartImages/'
    character = 'P00A+000E+00.pgm'
    # obj = LBP_Implement(R, P, type, uniform, w_num, h_num, overlap_size)
    # obj.run_LBP(path, character)
    # char = 'P00A+000E+20.pgm'
    # weights = obj.calculate_Weights(path,char)
    # print(weights)
    #
    # characters = array(['P00A+000E-20.pgm', 'P00A+000E-35.pgm', 'P00A+000E+20.pgm', 'P00A+000E+45.pgm'])
    # temp = 0.0
    # acc = 0.0
    # for char in characters:
    #     temp = obj.calculate_Accuracy(path,char)
    #     print('Recognition Rate: %-10.3f'%temp)
    #     acc += temp
    # print('Final rate: %-10.3f'%(acc/4))

    # 7*7
    # overlap_ratio 0.25, 6 pixels

    # characters = array(['P00A+000E-20.pgm', 'P00A+000E-35.pgm', 'P00A+000E+20.pgm', 'P00A+000E+45.pgm'])
    # overlap_ratios = array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4])
    # acc = [0.0] * 8
    # count = 0
    # for overlap_ratio in overlap_ratios:
    #     obj = LBP_Implement(R, P, type, uniform, w_num, h_num, overlap_ratio)
    #     obj.run_LBP(path, character)
    #     temp = 0.0
    #     for char in characters:
    #         temp += obj.calculate_Accuracy(path, char)
    #     acc[count] = temp / 4
    #     count +=1
    # plt.figure()
    # plt.plot(overlap_ratios, acc, c='c', ls='-.', linewidth=3)
    # plt.xlabel('Overlap Size')
    # plt.ylabel('Mean Recognition Rate')
    # plt.show()
