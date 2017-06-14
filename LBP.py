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
    obj = LBP_Implement(R, P, type, uniform, w_num, h_num, overlap_ratio)
    obj.run_LBP(path, character)
    # char = 'P00A+000E+20.pgm'
    # weights = obj.calculate_Weights(path,char)
    # print(weights)
    # #
    characters = array(['P00A+000E-20.pgm', 'P00A+000E-35.pgm', 'P00A+000E+20.pgm', 'P00A+000E+45.pgm','P00A+020E-40.pgm'])
    temp = 0.0
    acc = 0.0
    for char in characters:
        temp = obj.calculate_Accuracy(path,char)
        print('Recognition Rate: %-10.3f'%temp)
        acc += temp
    print('Final rate: %-10.3f'%(acc/5))

