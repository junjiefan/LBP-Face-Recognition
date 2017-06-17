from LBP_Implement import LBP_Implement
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    R = 1
    P = 8
    type = 1  # 1 for original, 0 for circular neighbor-sets
    uniform = 1  # 1: use uniform patterns, 0: not
    w_num = 7
    h_num = 7
    overlap_ratio = 0  # from 0 to 1
    path = '/cs/home/jf231/Dissertation/CS5099/PartImages/'
    #path = 'F:/dissertation/Images/'
    hori_angle = '+000E'
    ver_angle = '+00'
    obj = LBP_Implement(R, P, type, uniform, w_num, h_num, overlap_ratio)
    obj.select_Features(path,hori_angle,'+00','+20')
    # obj.run_LBP(path, hori_angle,ver_angle)
    # weights = obj.calculate_Weights(path,hori_angle,'+20')
    # print(weights.reshape(7,7))
    # characters = array(['P00A+000E+00.pgm','P00A+000E-20.pgm', 'P00A+000E-35.pgm', 'P00A+000E+20.pgm', 'P00A+000E+45.pgm','P00A+000E+90.pgm'])
    # temp = 0.0
    # acc = 0.0
    # for char in characters:
    #     temp = obj.calculate_Accuracy(path,char)
    #     print('Recognition Rate: %-10.3f'%temp)
    #     acc += temp
    # print('Final rate: %-10.3f'%(acc/6))
    # weights = []
    # horizon_angles = np.array(['+000E', '+005E', '+010E', '+015E', '+020E', '+025E'])
    # vertical_angles = np.array(['+00', '+20', '-20', '-35', '+45', '+90'])
    # columns = ('+00', '+20', '-20', '-35', '+45', '+90')
    # rows = columns
    # length = len(columns)
    # colors = plt.cm.BuPu(np.linspace(0, 0.5, length))
    # table_content = []
    # obj = LBP_Implement(R, P, type, uniform, w_num, h_num, overlap_ratio)
    # for i in range(length):
    #     print(vertical_angles[i])
    #     acc = np.array([0.0] * length)
    #     obj.run_LBP(path, horizon_angles[0], vertical_angles[i])
    #     for j in range(length):
    #         accuracy = obj.calculate_Accuracy(path,horizon_angles[0],vertical_angles[j],weights)
    #         acc[j] = np.around(accuracy*100, decimals=2)
    #     print(acc)
    #     table_content.append(acc)
    # ax = plt.subplot(frame_on=False)
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    # the_table = plt.table(cellText=table_content,
    #                       rowLabels=rows,
    #                       colLabels=columns,
    #                       rowColours=colors,
    #                       colColours = colors,
    #                       colWidths=[.15] * length,
    #                       loc='center')
    # the_table.auto_set_font_size(False)
    # the_table.set_fontsize(13)
    # the_table.scale(1, 2)
    # plt.title("Accuracy under condition \'+000E\'")
    # plt.show()