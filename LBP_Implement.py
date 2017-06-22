import os
import math
from numpy import *
import cv2
import time
from functools import wraps


class LBP_Implement(object):
    # R: the radius, P: the number of sampling points, type: original LBP or circular LBP operator
    # uniform: uniform patterns, devided into w_num * h_num regions
    # overlap_ratio: the ratio of overlapping regions
    def __init__(self, R, P, type, uniform, w_num, h_num, overlap_ratio):
        self.Width = 168
        self.Height = 192
        self.Image_Num = 0
        self.Radius = R
        self.Points = P
        self.lbp_type = type
        self.uniform = uniform
        self.w_num = w_num
        self.h_num = h_num
        self.LBPoperator = 0
        self.Histograms = 0
        # self.weight = 0
        self.isweighted = 0
        self.ids = 0
        self.overlap_ratio = overlap_ratio
        self.gradients = 0
        self.gamma = 0.15
        if (self.overlap_ratio > 0):
            region_width = self.Width / self.w_num
            region_height = self.Height / self.h_num
            self.overlap_size = int(region_width * self.overlap_ratio)
            print('Overlap_size: %d' % self.overlap_size)
            self.region_w_num = 1 + math.ceil((self.Width - region_width) / (region_width - self.overlap_size))
            self.region_h_num = 1 + math.ceil((self.Height - region_height) / (region_height - self.overlap_size))
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
    def loadImages(self, mypath):
        FaceMat = mat(zeros((self.Image_Num,
                             self.Width * self.Height)))
        j = 0
        for m in os.listdir(mypath):
            if (len(m) == 24):
                try:
                    img = cv2.imread(mypath + m, 0)
                    self.ids[j] = int(m[5:7])
                    self.gradients[:, j] = self.cal_Gradient(mat(img))
                    # img = self.gamma_correction(img, self.gamma)
                    # img = cv2.GaussianBlur(img, (3, 3), 0)
                    # img = cv2.equalizeHist(img)
                    FaceMat[j, :] = mat(img).flatten()
                    j = j + 1
                except:
                    print('Load %s failed' % m)
        print('Successfully loaded %s images' % j)
        return FaceMat  # Rotate the binary string and obtain a minimal binary number for each pattern

    def gamma_correction(self, img, correction):
        img = img / 255.0
        img = cv2.pow(img, correction)
        return uint8(img * 255)

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
    def LBP(self, FaceMat, train):
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
            for x in range(R, H - R):
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
            # if train == 1:
            # self.gradients[:, i] = self.cal_Gradient(face)
            #     cv2.imwrite('/cs/home/jf231/Dissertation/CS5099/LBP_Images/' + str(i) + '.jpg', array(tempface, uint8))
            # print(tempface[0:10, 0:10])
        return LBPoperator

    # Utilize Sobel operator calculate image gradients.
    # THe gradient will be used as weight for each pixel
    def cal_Gradient(self, face):
        x64 = cv2.Sobel(face, cv2.CV_64F, 1, 0, ksize=5)
        y64 = cv2.Sobel(face, cv2.CV_64F, 0, 1, ksize=5)
        sobelx = uint8(absolute(x64))
        sobely = uint8(absolute(y64))
        res = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        max = amax(res)
        min = amin(res)
        normalized = around((res - min) / (max - min), decimals=4)
        # print(res[0:10, 0:10])
        res = normalized.flatten().T.reshape(self.Width * self.Height, 1)
        return res

    # Calculate the number of transitions in the binary pattern and judge whether it is a uniform pattern
    def transition_number(self, pattern):
        length = len(pattern)
        count = 0
        count = abs(int(pattern[length - 1]) - int(pattern[0]))
        for p in range(1, length):
            count += abs(int(pattern[p]) - int(pattern[p - 1]))
        return count

    # This function need to be optimized, is used in circular LBP operator
    def bilinear_interpolation(self, face, x, y):
        x1 = int(x)
        x2 = x1 + 1
        y1 = int(y)
        y2 = y1 + 1
        if ((y2 < self.Width) and (x2 < self.Height)):
            P11 = face[x1, y1]
            P12 = face[x1, y2]
            P21 = face[x2, y1]
            P22 = face[x2, y2]
            x = x - x1
            y = y - y1
            pixel = (1 - x) * (1 - y) * P11 + x * (1 - y) * P21 + (1 - x) * y * P12 + x * y * P22
            return int(pixel)
        else:
            return 0

    # Calculate histogram for each image
    def calHistogram(self, ImgLBPope, gradients):
        # All 59 uniform patterns, when P is 8.
        patterns = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96,
                    112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207,
                    223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255, 9]
        Img = ImgLBPope.reshape(self.Height, self.Width)  # Height: rows, Width: columns
        gradients = gradients.reshape(self.Height, self.Width)
        mask_height, mask_width = int(self.Height / self.h_num), int(self.Width / self.w_num)
        # Divide the image into local regions
        if self.overlap_ratio > 0:
            Histogram = mat(zeros((self.Patterns, self.region_w_num * self.region_h_num)))
            count = 0
            for i in range(self.region_h_num):
                for j in range(self.region_w_num):
                    mask = zeros(shape(Img), uint8)
                    start_x = i * mask_height - i * self.overlap_size
                    end_x = (i + 1) * mask_height - i * self.overlap_size
                    start_y = j * mask_width - j * self.overlap_size
                    end_y = (j + 1) * mask_width - j * self.overlap_size

                    if (end_x < self.Height and end_y < self.Width):
                        # mask[start_x:end_x, start_y:end_y] = 255
                        x1 = start_x;
                        x2 = end_x
                        y1 = start_y;
                        y2 = end_y
                    else:
                        if (end_x >= self.Height and end_y < self.Width):
                            x1 = self.Height - mask_height - 1;
                            y1 = start_y
                            x2 = self.Height - 1;
                            y2 = end_y
                            # mask[x1:x2, start_y:end_y] = 255
                        if (end_x < self.Height and end_y >= self.Width):
                            y1 = self.Width - mask_width - 1;
                            x1 = start_x
                            y2 = self.Width - 1;
                            x2 = end_x
                            mask[start_x:end_x, y1:y2] = 255
                        if (end_x >= self.Height and end_y >= self.Width):
                            x1 = self.Height - mask_height - 1
                            x2 = self.Height - 1
                            y1 = self.Width - mask_width - 1
                            y2 = self.Width - 1
                            # mask[x1:x2, y1:y2] = 255
                    # hist = cv2.calcHist([array(Img, uint8)], [0], mask, [self.Patterns], [0, 256])
                    hist = [0.0] * self.Patterns
                    for c in range(x1, x2):
                        for r in range(y1, y2):
                            pattern = Img[c, r]
                            for k in range(self.Patterns):
                                if pattern == patterns[k]:
                                    hist[k] += gradients[c, r]
                                    # hist[k] += 1
                    # The image; the channel; the mask; the number of bins; the range of value
                    Histogram[:, count] = mat(hist).flatten().T
                    count += 1
        else:
            Histogram = mat(zeros((self.Patterns, self.w_num * self.h_num)))
            # print(shape(Histogram))
            count = 0
            for i in range(self.h_num):
                for j in range(self.w_num):
                    hist = [0.0] * self.Patterns
                    for c in range(i * mask_height, (i + 1) * mask_height):
                        for r in range(j * mask_width, (j + 1) * mask_width):
                            pattern = Img[c, r]
                            for k in range(self.Patterns):
                                if pattern == patterns[k]:
                                    # hist[k] += gradients[c, r]
                                    # hist[k] += math.log((gradients[c, r] + 1), math.e)
                                    hist[k] += math.exp(gradients[c, r]) - 1
                                    # hist[k] += 1
                    Histogram[:, count] = mat(hist).flatten().T
                    count += 1
        # from sklearn.preprocessing import normalize
        # Histogram = mat(normalize(Histogram,norm='l1',axis=0))
        return Histogram.flatten().T

    # recogniseImg: the image needed to be matched
    # LBPoperator: LBP operators for all images
    # exHistograms: the calculated histograms for all image
    def recogniseFace(self, recogniseImg, weights, ImgGradient):
        recogniseImg = recogniseImg.T
        ImgLBPope = self.LBP(recogniseImg, 0)
        recogniseHistogram = self.calHistogram(ImgLBPope, ImgGradient)
        minIndex = 0
        minVals = inf
        # Find the most close one, the smallest difference
        for i in range(shape(self.LBPoperator)[1]):
            Histogram = self.Histograms[:, i]
            # Utilize Chi square distance to find the most close match

            if (self.isweighted == 0):
                distance = ((array(Histogram - recogniseHistogram) ** 2).sum()) / (
                    (array(Histogram + recogniseHistogram)).sum())
            else:
                if (self.overlap_ratio == 0):
                    regions = self.h_num * self.w_num
                else:
                    regions = self.region_h_num * self.region_w_num
                Histogram = Histogram.reshape(self.Patterns, regions)
                recogniseHistogram = recogniseHistogram.reshape(self.Patterns, regions)
                distance = 0
                for index in range(regions):
                    distance += (((array(Histogram[:, index] - recogniseHistogram[:, index])) ** 2).sum() / (
                        array(Histogram[:, index] + recogniseHistogram[:, index])).sum()) * weights[index]

            if distance < minVals:
                minIndex = i
                minVals = distance

        return minIndex

    @fn_timer
    def run_LBP(self, path):
        self.Image_Num = len([file for file in os.listdir(path)
                              if os.path.isfile(os.path.join(path, file))])
        self.ids = [0 for i in range(self.Image_Num)]
        self.gradients = mat(zeros((self.Width * self.Height, self.Image_Num)))
        # Load images
        FaceMat = self.loadImages(path).T
        # Calculate LBP opearters for all loaded images
        self.LBPoperator = self.LBP(FaceMat, 1)
        # Calculate histograms
        if self.overlap_ratio > 0:
            self.Histograms = mat(
                zeros((self.Patterns * self.region_w_num * self.region_h_num, shape(self.LBPoperator)[1])))
        else:
            self.Histograms = mat(zeros((self.Patterns * self.w_num * self.h_num, shape(self.LBPoperator)[1])))
        for i in range(shape(self.LBPoperator)[1]):
            Histogram = self.calHistogram(self.LBPoperator[:, i], self.gradients[:, i])
            self.Histograms[:, i] = Histogram

    def calculate_Accuracy(self, mypath, weights):
        if (len(weights) > 1):
            self.isweighted = 1
        j = 0
        count = 0
        for m in os.listdir(mypath):
            if (len(m) == 24):
                recogniseImg = cv2.imread(mypath + m, 0)
                ImgGradient = self.cal_Gradient(mat(recogniseImg))
                # recogniseImg = self.gamma_correction(recogniseImg, self.gamma)
                # recogniseImg = cv2.GaussianBlur(recogniseImg, (3, 3), 0)
                # recogniseImg = cv2.equalizeHist(recogniseImg)
                id = int(m[5:7])
                index = self.recogniseFace(mat(recogniseImg).flatten(), weights, ImgGradient)
                if self.ids[index] == id:
                    count = count + 1
                j = j + 1
        if j > 0:
            accuracy = float(count) / j
            return accuracy
        else:
            return 0

    def select_Features(self, mypath, hori_angle, cons, subject_num):
        con_num = len(cons)
        print(con_num)
        intra_num = int((subject_num * con_num * (con_num - 1)) / 2)
        extra_num = int((subject_num * (subject_num - 1) * con_num))
        print('intra %d' % intra_num)
        print('extra %d' % extra_num)
        if (self.overlap_ratio == 0):
            my_rows = self.h_num
            my_columns = self.w_num
        else:
            my_rows = self.region_h_num
            my_columns = self.region_w_num
        region_num = my_rows * my_columns
        intra_distance = mat(zeros((intra_num, region_num)))
        intra_y = array([1 for i in range(intra_num)])
        extra_distance = mat(zeros((extra_num, region_num)))
        extra_y = array([0 for i in range(extra_num)])
        extra_index = 0
        intra_index = 0
        # load images, intra-personal pairs and extra-personal pairs
        for con_index in range(con_num):
            for i in os.listdir(mypath):
                if (len(i) == 24):
                    id1 = int(i[5:7])
                    h1 = i[12:17]
                    v1 = i[17:20]
                    if (h1 == hori_angle) and (v1 == cons[con_index]):
                        # print('The image %s' % i)
                        img1 = cv2.imread(mypath + i, 0)
                        # img1 = self.gamma_correction(img1, self.gamma)
                        lbp_op1 = self.LBP(mat(img1).flatten().T, 0)
                        img_gra1 = self.cal_Gradient(mat(img1).flatten().T)
                        hist1 = self.calHistogram(lbp_op1, img_gra1)
                        hist1 = hist1.reshape(self.Patterns, region_num)
                        # Read another image to form a pair
                        for j in os.listdir(mypath):
                            if (len(j) == 24):
                                id2 = int(j[5:7])
                                h2 = j[12:17]
                                v2 = j[17:20]
                                if (h2 == hori_angle) and (v2 == cons[(con_index + 1) % con_num]):
                                    img2 = cv2.imread(mypath + j, 0)
                                    # img2 = self.gamma_correction(img2, self.gamma)
                                    lbp_op2 = self.LBP(mat(img2).flatten().T, 0)
                                    img_gra2 = self.cal_Gradient(mat(img2).flatten().T)
                                    hist2 = self.calHistogram(lbp_op2, img_gra2)
                                    hist2 = hist2.reshape(self.Patterns, region_num)
                                    for region_index in range(region_num):
                                        region_1 = hist1[:, region_index]
                                        region_2 = hist2[:, region_index]
                                        para1 = (array(region_1 - region_2) ** 2).sum()
                                        para2 = (array(region_1 + region_2)).sum()
                                        distance = para1 / para2
                                        if (id1 == id2):
                                            # two images belong to the same person
                                            intra_distance[intra_index, region_index] = distance
                                        else:
                                            # two images belong to different persons
                                            extra_distance[extra_index, region_index] = distance
                                    if (id1 == id2):
                                        intra_index += 1
                                    else:
                                        extra_index += 1
        print(intra_index)
        print(extra_index)
        # print(intra_distance[0:5,0:5])
        # print(extra_distance[0:5,0:5])

        intra = concatenate((intra_distance, intra_y.reshape(shape(intra_y)[0], 1)), axis=1)
        extra = concatenate((extra_distance, extra_y.reshape(shape(extra_y)[0], 1)), axis=1)
        import pandas as pd
        intra = pd.DataFrame(intra)
        intra.to_csv('intra.csv', sep=',', index=False)
        extra = pd.DataFrame(extra)
        extra.to_csv('extra.csv', sep=',', index=False)
        from FeatureSelection import feature_Select
        fs = feature_Select(intra_distance, extra_distance, intra_y, extra_y)
        return fs

    # This is a linear function to calculate weight for each region
    @fn_timer
    def calculate_Weights(self, mypath):
        image_num = len([file for file in os.listdir(mypath)
                         if os.path.isfile(os.path.join(mypath, file))])
        self.isweighted = 1
        if (self.overlap_ratio == 0):
            my_rows = self.h_num
            my_columns = self.w_num
        else:
            my_rows = self.region_h_num
            my_columns = self.region_w_num

        weights = mat(zeros((image_num, my_rows * my_columns)))
        rows, columns = shape(weights)
        count = 0
        for m in os.listdir(mypath):
            if (len(m) == 24):
                image = cv2.imread(mypath + m, 0)
                # imgage = self.gamma_correction(imgage, self.gamma)
                id = int(m[5:7])
                imageLBP = self.LBP(mat(image).flatten().T, 0)
                imageGradient = self.cal_Gradient(mat(image).flatten().T)
                histogram = self.calHistogram(imageLBP, imageGradient)
                histogram = histogram.reshape(self.Patterns, columns)
                temp = [0] * columns
                for i in range(columns):
                    # calculate the recognition rate for each region
                    local_region = histogram[:, i]
                    min_index = 0
                    min_value = inf
                    for j in range(self.Image_Num):
                        stored_hist = self.Histograms[:, j]
                        stored_hist = stored_hist.reshape(self.Patterns, columns)
                        stored_region = stored_hist[:, i]
                        para1 = (array(local_region - stored_region) ** 2).sum()
                        para2 = (array(local_region + stored_region)).sum()
                        distance = para1 / para2
                        if (distance < min_value):
                            min_index = j
                            min_value = distance
                    match_id = self.ids[min_index]
                    if (id == match_id):
                        temp[i] = 1
                    else:
                        temp[i] = 0
                weights[count, :] = temp
                count += 1

        weights = weights.mean(axis=0)
        temp = array([0.0] * columns)
        for n in range(columns):
            temp[n] = weights[0, n]

        temp = temp.reshape(my_rows, my_columns)
        # print(temp)
        H, W = shape(temp)
        adjust = int(W / 2)
        for row in range(H):
            for col in range(adjust):
                average = (temp[row][col] + temp[row][W - col - 1]) / 2
                temp[row][col] = average
                temp[row][W - col - 1] = average
        temp = temp.flatten()
        sorted_weights = sorted(temp)
        thresholds = [0.1, 0.8, 0.9, 1]
        weight_standard = [0, 1, 2, 4]
        start = -1
        for t in range(4):
            index = int(columns * thresholds[t]) - 1
            end = sorted_weights[index]
            for j in range(columns):
                if ((temp[j] <= end) and (temp[j] > start)):
                    temp[j] = weight_standard[t]
            start = end
        weights = temp
        # weights = temp.reshape(my_rows, my_columns)
        return weights
