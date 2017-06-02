import os
import math
from numpy import *
import cv2
import time
from functools import wraps


class LBP_Implement(object):
    def __init__(self, R, P, type, uniform, w_num, h_num, overlap_size):
        self.Width = 168
        self.Height = 192
        self.Image_Num = 10
        self.Radius = R
        self.Points = P
        self.lbp_type = type
        self.uniform = uniform
        self.w_num = w_num
        self.h_num = h_num
        self.LBPoperator = 0
        self.Histograms = 0
        self.ids = ids = [0 for i in range(self.Image_Num)]
        self.overlap_size = overlap_size
        if (self.uniform == 1):
            self.Patterns = self.Points * (self.Points - 1) + 3
            # P*(P-1) for patterns with two transitions,
            # 2 bins for patterns with zero transitions,
            # 1 bin for all miscellaneous patterns
        else:
            self.Patterns = 2 ** self.Points

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
    def loadImages(self, mypath, character):
        FaceMat = mat(zeros((self.Image_Num,
                             self.Width * self.Height)))  # Establish a matrix, first parameter is the number of images, second one is the resolution
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
                        self.ids[j] = int(m[5:7])
                    except:
                        print('Load %s failed' % m)
                    FaceMat[j, :] = mat(img).flatten()
                    j = j + 1
        print('Successfully loaded %s images' % j)
        return FaceMat  # Rotate the binary string and obtain a minimal binary number for each pattern

    # Start from the end of the binary string, remove all '0' at the end and place them in the begining.
    def minBinary(self, pixel):
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
    def LBP(self, FaceMat, save):
        R = self.Radius
        # The offset of 3*3 neighbors
        if (self.lbp_type == 1):
            Neighbor_x = [-1, 0, 1, 1, 1, 0, -1, -1]
            Neighbor_y = [-1, -1, -1, 0, 1, 1, 1, 0]
        else:
            pi = math.pi
        LBPoperator = mat(zeros(shape(FaceMat)))  # shape tells the number of row and column of the matrix
        for i in range(shape(FaceMat)[1]):  # obtain the number of column
            face = FaceMat[:, i].reshape(self.Height, self.Width)  # Height represents the number of row
            H, W = shape(face)
            tempface = mat(zeros((H, W)))
            for x in range(R, H - R):  # range(1,5),represents 1 to 4, which is 1,2,3,4
                for y in range(R, W - R):
                    repixel = ''
                    pixel = int(face[x, y])
                    # Original LBP algorithm, 3*3
                    if (self.lbp_type == 1):
                        for p in range(8):
                            xp = x + Neighbor_x[p]
                            yp = y + Neighbor_y[p]
                            if int(face[xp, yp]) > pixel:
                                repixel += '1'
                            else:
                                repixel += '0'
                    # Utilize circular regions, with changeable radius and number of points
                    else:
                        for p in [3, 4, 5, 6, 7, 0, 1, 2]:
                            p = float(p)
                            xp = x + R * cos(2 * pi * (p / self.Points))
                            yp = y - R * sin(2 * pi * (p / self.Points))
                            neighbor_pixel = self.bilinear_interpolation(face, xp, yp)
                            if neighbor_pixel > pixel:
                                repixel += '1'
                            else:
                                repixel += '0'
                                # Obtain the minimal binary string for each pattern
                                # tempface[x, y] = minBinary(repixel)
                    if (self.uniform == 1):
                        U = self.transition_number(repixel)
                        if U <= 2:
                            tempface[x, y] = int(repixel, base=2)
                        else:
                            tempface[x, y] = self.Points + 1
                    else:
                        tempface[x, y] = int(repixel, base=2)
            LBPoperator[:, i] = tempface.flatten().T
            if save == 1:
                cv2.imwrite('/cs/home/jf231/Dissertation/CS5099/LBP_Images/' + str(i) + '.jpg', array(tempface, uint8))
        return LBPoperator

    # Calculate the number of transitions in the binary pattern and judge whether it is a uniform pattern
    def transition_number(self, pattern):
        length = len(pattern)
        count = 0
        count = abs(int(pattern[length - 1]) - int(pattern[0]))
        for p in range(1, length):
            count += abs(int(pattern[p]) - int(pattern[p - 1]))
        return count

    # This function need to be optimized
    def bilinear_interpolation(self, face, x, y):
        x1 = int(x)
        x2 = x1 + 1
        y1 = int(y)
        y2 = y1 + 1
        if ((y2 < self.Width) & (x2 < self.Height)):
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
    def calHistogram(self, ImgLBPope):
        Img = ImgLBPope.reshape(self.Height, self.Width)  # Height: rows, Width: columns
        # rows, columns = shape(Img)
        mask_height, mask_width = int(self.Height / self.h_num), int(self.Width / self.w_num)
        # Divide the image into local regions
        if self.overlap_size > 0:
            region_w_num = math.ceil(self.Width / (self.Width / self.w_num - self.overlap_size))
            region_h_num = math.ceil(self.Height / (self.Height / self.h_num - self.overlap_size))
            Histogram = mat(zeros((self.Patterns, region_w_num * region_h_num)))
            for i in range(region_h_num):
                for j in range(region_w_num):
                    mask = zeros(shape(Img), uint8)
                    start_x = i * mask_height - i * self.overlap_size
                    end_x = (i + 1) * mask_height - i * self.overlap_size
                    start_y = j * mask_width - j * self.overlap_size
                    end_y = (j + 1) * mask_width - j * self.overlap_size

                    if (end_x < self.Height and end_y < self.Width):
                        mask[start_x:end_x, start_y:end_y] = 255
                    else:
                        if (end_x >= self.Height and end_y < self.Width):
                            x1 = self.Height - mask_height - 1
                            x2 = self.Height - 1
                            mask[x1:x2, start_y:end_y] = 255
                        # mask[start_x:(self.Height - 1), start_y:end_y] = 255
                        # mask[0:end_x - self.Height + 1, start_y:end_y] = 255
                        if (end_x < self.Height and end_y >= self.Width):
                            y1 = self.Width - mask_width - 1
                            y2 = self.Width - 1
                            mask[start_x:end_x, y1:y2] = 255
                        # mask[start_x:end_x, start_y:(self.Width - 1)] = 255
                        # mask[start_x:end_x, 0:end_y - self.Width + 1] = 255
                        if (end_x >= self.Height and end_y >= self.Width):
                            x1 = self.Height - mask_height - 1
                            x2 = self.Height - 1
                            y1 = self.Width - mask_width - 1
                            y2 = self.Width - 1
                            mask[start_x:end_x, y1:y2] = 255
                    # mask[start_x:(self.Height - 1), start_y:(self.Width - 1)] = 255
                    hist = cv2.calcHist([array(Img, uint8)], [0], mask, [self.Patterns], [0, 256])
                    # The image; the channel; the mask; the number of bins; the range of value
                    Histogram[:, (i + 1) * (j + 1) - 1] = mat(hist).flatten().T
        else:
            Histogram = mat(zeros((self.Patterns, self.w_num * self.h_num)))
            for i in range(self.h_num):
                for j in range(self.w_num):
                    # unit8: unsigned integer, 0 - 255
                    mask = zeros(shape(Img), uint8)
                    mask[i * mask_height: (i + 1) * mask_height, j * mask_width:(j + 1) * mask_width] = 255
                    hist = cv2.calcHist([array(Img, uint8)], [0], mask, [self.Patterns], [0, 256])
                    # The image; the channel; the mask; the number of bins; the range of value
                    Histogram[:, (i + 1) * (j + 1) - 1] = mat(hist).flatten().T
        return Histogram.flatten().T

    # recogniseImg: the image needed to be matched
    # LBPoperator: LBP operators for all images
    # exHistograms: the calculated histograms for all image
    def recogniseFace(self, recogniseImg):
        recogniseImg = recogniseImg.T
        ImgLBPope = self.LBP(recogniseImg, 0)
        recongniseHistogram = self.calHistogram(ImgLBPope)
        minIndex = 0
        minVals = inf
        # inf stands for infinity, a value that is greater than any other value
        # Find the most close one, the smallest difference
        for i in range(shape(self.LBPoperator)[1]):
            Histogram = self.Histograms[:, i]
            # Utilize Chi square distance to find the most close match
            distance = ((array(Histogram - recongniseHistogram) ** 2).sum()) / (
                (array(Histogram + recongniseHistogram)).sum())
            if distance < minVals:
                minIndex = i
                minVals = distance
        return minIndex

    @fn_timer
    def run_LBP(self, path, character):
        # Load images
        FaceMat = self.loadImages(path, character).T
        # Calculate LBP opearters for all loaded images
        self.LBPoperator = self.LBP(FaceMat, 1)
        # Calculate histograms
        if self.overlap_size > 0:
            region_w_num = math.ceil(self.Width / (self.Width / self.w_num - self.overlap_size))
            region_h_num = math.ceil(self.Height / (self.Height / self.h_num - self.overlap_size))
            self.Histograms = mat(zeros((self.Patterns * region_w_num * region_h_num, shape(self.LBPoperator)[1])))
        else:
            self.Histograms = mat(zeros((self.Patterns * self.w_num * self.h_num, shape(self.LBPoperator)[1])))

        for i in range(shape(self.LBPoperator)[1]):
            Histogram = self.calHistogram(self.LBPoperator[:, i])
            self.Histograms[:, i] = Histogram

    def calculate_Accuracy(self, mypath, character):
        j = 0
        count = 0
        for m in os.listdir(mypath):
            str = m.split('_')
            if (len(str) == 2):
                if str[1] == character:
                    recogniseImg = cv2.imread(mypath + m, 0)
                    id = int(m[5:7])
                    index = self.recogniseFace(mat(recogniseImg).flatten())
                    if self.ids[index] == id:
                        count = count + 1
                    j = j + 1
        # print('Total: %d images'%j)
        # print('%d images mathched' % count)
        if j > 0:
            accuracy = float(count) / j
            return accuracy
        else:
            return 0
