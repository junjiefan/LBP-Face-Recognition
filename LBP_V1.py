import os
import math
from numpy import *
import cv2

Width = 168
Height = 192
Image_Num = 10
Points = 0
Patterns = 0
LBPoperator = 0
Histograms = 0
Radius = 0
lbp_type = 1
win_num = 4

import time
from functools import wraps


# Used for calculating consumed time
def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Running %s: %s seconds" % (function.__name__, str(t1 - t0)))
        return result

    return function_timer


# Load all images, according to the structure of CroppedYale,
# each folder contains several images of one person, and totally, 38 folders
def loadImages(mypath, character):
    FaceMat = mat(zeros((Image_Num,
                         Width * Height)))  # Establish a matrix, first parameter is the number of images, second one is the resolution
    j = 0
    # for m in os.listdir(mypath):
    #     for n in os.listdir(os.path.join(mypath, m)):
    #         if (len(n.split('_')) == 2):
    #             if n.split('_')[1] == character:

    for m in os.listdir(mypath):
        str = m.split('_')
        if (len(str) == 2):
            if str[1] == character:
                try:
                    img = cv2.imread(mypath + m, 0)
                except:
                    print('Load %s failed' % m)
                FaceMat[j, :] = mat(img).flatten()
                j = j + 1
    print('Successfully loaded %s images' % j)
    return FaceMat


# Rotate the binary string and obtain a minimal binary number for each pattern
# Start from the end of the binary string, remove all '0' at the end and place them in the begining.
def minBinary(pixel):
    length = len(pixel)
    last = length - 1
    result = inf
    for i in range(length):
        p = pixel[last]
        pixel = p + pixel[:last]
        temp = int(pixel, base=2)
        if (temp < result):
            result = temp
    return result


# Main component of LBP
def LBP(FaceMat):
    R = Radius
    # The offset of 3*3 neighbors
    if (lbp_type == 1):
        Neighbor_x = [-1, 0, 1, 1, 1, 0, -1, -1]
        Neighbor_y = [-1, -1, -1, 0, 1, 1, 1, 0]
    else:
        pi = math.pi
    LBPoperator = mat(zeros(shape(FaceMat)))  # shape tells the number of row and column of the matrix
    for i in range(shape(FaceMat)[1]):  # obtain the number of column
        face = FaceMat[:, i].reshape(Height, Width)  # Height represents the number of row
        H, W = shape(face)
        tempface = mat(zeros((H, W)))
        for x in range(R, H - R):  # range(1,5),represents 1 to 4, which is 1,2,3,4
            for y in range(R, W - R):
                repixel = ''
                pixel = int(face[x, y])
                # Original LBP algorithm, 3*3
                if (lbp_type == 1):
                    for p in range(8):
                        xp = x + Neighbor_x[p]
                        yp = y + Neighbor_y[p]
                        if int(face[xp, yp]) > pixel:
                            repixel += '1'
                        else:
                            repixel += '0'
                    tempface[x, y] = int(repixel, base=2)
                # Utilize circular regions, with changeable radius and number of points
                else:
                    for p in [3, 4, 5, 6, 7, 0, 1, 2]:
                        p = float(p)
                        xp = x + R * cos(2 * pi * (p / Points))
                        yp = y - R * sin(2 * pi * (p / Points))
                        neighbor_pixel = bilinear_interpolation(face, xp, yp)
                        if neighbor_pixel > pixel:
                            repixel += '1'
                        else:
                            repixel += '0'
                    # Obtain the minimal binary string for each pattern
                    # tempface[x, y] = minBinary(repixel)
                    U = transition_number(repixel)
                    if U <= 2:
                        tempface[x, y] = int(repixel, base=2)
                    else:
                        tempface[x, y] = Points + 1
        LBPoperator[:, i] = tempface.flatten().T
        cv2.imwrite('F:/dissertation/LBP_Images/' + str(i) + '.jpg', array(tempface, uint8))
    return LBPoperator


# Calculate the number of transitions in the binary pattern and judge whether it is a uniform pattern
def transition_number(pattern):
    length = len(pattern)
    count = 0
    count = abs(int(pattern[length - 1]) - int(pattern[0]))
    for p in range(1, length):
        count += abs(int(pattern[p]) - int(pattern[p - 1]))
    return count


# This function need to be optimized
def bilinear_interpolation(face, x, y):
    x1 = int(x)
    x2 = x1 + 1
    y1 = int(y)
    y2 = y1 + 1
    if ((y2 < Width) & (x2 < Height)):
        P11 = face[x1, y1]
        P12 = face[x1, y2]
        P21 = face[x2, y1]
        P22 = face[x2, y2]
        x = x - x1
        y = y - y1
        pixel = (1 - x) * (1 - y) * P11 + x * (1 - y) * P21 + (1 - x) * y * P12 + x * y * P22
        # pixel = (((x2 - x) * (y2 - y))/((x2 - x1) * (y2 - y1))) * P11 + (((x - x1) * (y2 - y)) / ((x2 - x1) * (y2 - y1))) * P21\
        #         + (((x2 - x) * (y - y1)) / ((x2 - x1) * (y2 - y1))) * P12 + (((x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))) * P22
        return int(pixel)
    else:
        return 0


# Calculate histogram for each image
def calHistogram(ImgLBPope):
    Img = ImgLBPope.reshape(Height, Width)  # Height: rows, Width: columns
    rows, columns = shape(Img)
    # Divide the image into local regions
    Histogram = mat(zeros((Patterns, win_num * win_num)))
    maskx, masky = int(rows / win_num), int(columns / win_num)
    for i in range(win_num):
        for j in range(win_num):
            # unit8: unsigned integer, 0 - 255
            mask = zeros(shape(Img), uint8)
            mask[i * maskx: (i + 1) * maskx, j * masky:(j + 1) * masky] = 255
            hist = cv2.calcHist([array(Img, uint8)], [0], mask, [Patterns], [0, 256])
            # The image; the channel; the mask; the number of bins; the range of value
            Histogram[:, (i + 1) * (j + 1) - 1] = mat(hist).flatten().T
    return Histogram.flatten().T


# recogniseImg: the image needed to be matched
# LBPoperator: LBP operators for all images
# exHistograms: the calculated histograms for all image
def recogniseFace(recogniseImg):
    recogniseImg = recogniseImg.T
    ImgLBPope = LBP(recogniseImg)
    recongniseHistogram = calHistogram(ImgLBPope)
    minIndex = 0
    minVals = inf
    # inf stands for infinity, a value that is greater than any other value
    # Find the most close one, the smallest difference
    for i in range(shape(LBPoperator)[1]):
        Histogram = Histograms[:, i]
        # Utilize Chi square distance to find the most close match
        distance = ((array(Histogram - recongniseHistogram) ** 2).sum()) / (
            (array(Histogram + recongniseHistogram)).sum())
        if distance < minVals:
            minIndex = i
            minVals = distance
    return minIndex


@fn_timer
def run_LBP(path, character, type, uniform, radius, points,win):
    global Points
    Points = points
    global lbp_type
    lbp_type = type
    global Radius
    Radius = radius
    global win_num
    win_num = win
    global Patterns
    if (uniform == 1):
        Patterns = Points * (Points - 1) + 3
        # P*(P-1) for patterns with two transitions,
        # 2 bins for patterns with zero transitions,
        # 1 bin for all miscellaneous patterns
    else:
        Patterns = 2 ** Points
    # Load images
    FaceMat = loadImages(path, character).T
    # Calculate LBP opearters for all loaded images
    global LBPoperator
    LBPoperator = LBP(FaceMat)
    # Calculate histogra
    global Histograms
    Histograms = mat(zeros((Patterns * win_num * win_num, shape(LBPoperator)[1])))
    for i in range(shape(LBPoperator)[1]):
        Histogram = calHistogram(LBPoperator[:, i])
        Histograms[:, i] = Histogram


def calculate_Accuracy(mypath, character):
    j = 0
    count = 0
    for m in os.listdir(mypath):
        str = m.split('_')
        if (len(str) == 2):
            if str[1] == character:
                recogniseImg = cv2.imread(mypath + m, 0)
                if recogniseFace(mat(recogniseImg).flatten()) == j:
                    count = count + 1
                j = j + 1
    accuracy = float(count) / j
    return accuracy
    # print('The accuracy of %s is' % character)
    # print(accuracy)
