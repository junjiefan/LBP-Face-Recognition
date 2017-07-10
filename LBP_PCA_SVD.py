import os
import math
import numpy as np
import cv2
import time
from functools import wraps


class LBP_PCA(object):
    # R: the radius, P: the number of sampling points, type: original LBP or circular LBP operator
    # uniform: uniform patterns, devided into w_num * h_num regions
    # overlap_ratio: the ratio of overlapping regions
    def __init__(self, R, P, type, uniform, w_num, h_num, overlap_ratio, gap_size):
        self.Width = 168
        self.Height = 192
        self.Radius = R
        self.Points = P
        self.lbp_type = type
        self.uniform = uniform
        self.w_num = w_num
        self.h_num = h_num
        self.overlap_ratio = overlap_ratio
        self.gamma = 0.15
        self.gap_size = gap_size
        self.weights = []
        if (self.gap_size > 0):
            self.Height = self.Height - self.gap_size * 2
            self.Width = self.Width - self.gap_size * 2
        if (self.overlap_ratio > 0):
            region_width = self.Width / self.w_num
            region_height = self.Height / self.h_num
            self.overlap_size = int(region_width * self.overlap_ratio)
            print('Overlap_size: %d' % self.overlap_size)
            self.region_w_num = 1 + math.ceil((self.Width - region_width) / (region_width - self.overlap_size))
            self.region_h_num = 1 + math.ceil((self.Height - region_height) / (region_height - self.overlap_size))
            # print(self.region_h_num)
            # print(self.region_w_num)
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

    # Load all image
    def loadImages(self, image_num, mypath, shift_vertical, shift_horizonal):
        FaceMat = np.mat(np.zeros((image_num,
                                   self.Width * self.Height)))
        ids = [0 for i in range(image_num)]
        gradients = np.mat(np.zeros((image_num, self.Width * self.Height)))
        j = 0
        for m in os.listdir(mypath):
            if (len(m) == 24):
                try:
                    img = cv2.imread(mypath + m, 0)
                    ids[j] = int(m[5:7])
                    img = np.mat(img)[
                          self.gap_size + shift_vertical:self.gap_size + shift_vertical + self.Height,
                          self.gap_size + shift_horizonal: self.gap_size + shift_horizonal + self.Width]
                    gradients[j, :] = self.cal_Gradient(np.mat(img))
                    # img = self.gamma_correction(img, self.gamma)
                    # img = cv2.GaussianBlur(img, (3, 3), 0)
                    # img = cv2.equalizeHist(img)
                    FaceMat[j, :] = np.mat(img).flatten()
                    j = j + 1
                except:
                    print('Load %s failed' % m)
        print('Successfully loaded %s images' % j)
        return ids, FaceMat, gradients

    def gamma_correction(self, img, correction):
        img = img / 255.0
        img = cv2.pow(img, correction)
        return np.uint8(img * 255)

    # Start from the end of the binary string, remove all '0' at the end and place them in the begining.
    def minBinary(self, pixel):
        length = len(pixel)
        last = length - 1
        result = np.inf
        for i in range(length):
            p = pixel[last]
            pixel = p + pixel[:last]
            temp = int(pixel, base=2)
            if (temp < result):
                result = temp
        return result

    # Main component of LBP
    def LBP(self, FaceMat):
        R = self.Radius
        # The offset of 3*3 neighbors
        if (self.lbp_type == 1):
            Neighbor_h = [-1, 0, 1, 1, 1, 0, -1, -1]
            Neighbor_w = [-1, -1, -1, 0, 1, 1, 1, 0]
        else:
            pi = math.pi
        LBPoperator = np.mat(np.zeros(np.shape(FaceMat)))
        for i in range(np.shape(FaceMat)[0]):  # obtain the number of images
            face = FaceMat[i, :].reshape(self.Height, self.Width)  # Height represents the number of row
            H, W = np.shape(face)
            tempface = np.mat(np.zeros((H, W)))
            for h in range(R, H - R):
                for w in range(R, W - R):
                    repixel = ''
                    pixel = int(face[h, w])
                    # Original LBP algorithm, 3*3
                    if (self.lbp_type == 1):
                        for p in range(8):
                            hp = h + Neighbor_h[p]
                            wp = w + Neighbor_w[p]
                            if int(face[hp, wp]) > pixel:
                                repixel += '1'
                            else:
                                repixel += '0'
                    # Utilize circular regions, with changeable radius and number of points
                    else:
                        for p in [3, 4, 5, 6, 7, 0, 1, 2]:
                            p = float(p)
                            hp = h + R * np.cos(2 * pi * (p / self.Points))
                            wp = w - R * np.sin(2 * pi * (p / self.Points))
                            neighbor_pixel = self.bilinear_interpolation(face, hp, wp)
                            if neighbor_pixel > pixel:
                                repixel += '1'
                            else:
                                repixel += '0'
                                # Obtain the minimal binary string for each pattern
                                # tempface[x, y] = minBinary(repixel)
                    if (self.uniform == 1):
                        U = self.transition_number(repixel)
                        if U <= 2:
                            tempface[h, w] = int(repixel, base=2)
                        else:
                            tempface[h, w] = self.Points + 1
                    else:
                        tempface[h, w] = int(repixel, base=2)
            LBPoperator[i, :] = tempface.flatten()
        return LBPoperator

    # Utilize Sobel operator calculate image gradients.
    # THe gradient will be used as weight for each pixel
    def cal_Gradient(self, face):
        x64 = cv2.Sobel(face, cv2.CV_64F, 1, 0, ksize=5)
        y64 = cv2.Sobel(face, cv2.CV_64F, 0, 1, ksize=5)
        sobelx = np.uint8(np.absolute(x64))
        sobely = np.uint8(np.absolute(y64))
        res = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        row_sums = res.sum(axis=1)
        res = np.around(res / row_sums[:, np.newaxis], decimals=4)
        if self.exp_para != 0:
            H, W = np.shape(res)
            for i in range(H):
                for j in range(W):
                    # res[i, j] = around(math.log(100 * res[i, j] + 1), decimals=4)
                    res[i, j] = np.around(math.exp(self.exp_para * res[i, j]) - 1, decimals=4)
        res = res.flatten().reshape(1, self.Width * self.Height)
        return res

    # Calculate the number of transitions in the binary pattern and judge whether it is a uniform pattern
    def transition_number(self, pattern):
        length = len(pattern)
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
            Histogram = np.mat(np.zeros((self.Patterns, self.region_w_num * self.region_h_num)))
            count = 0
            for i in range(self.region_h_num):
                for j in range(self.region_w_num):
                    start_h = i * mask_height - i * self.overlap_size
                    end_h = (i + 1) * mask_height - i * self.overlap_size
                    start_w = j * mask_width - j * self.overlap_size
                    end_w = (j + 1) * mask_width - j * self.overlap_size

                    if (end_h < self.Height and end_w < self.Width):
                        # mask[start_x:end_x, start_y:end_y] = 255
                        h1 = start_h
                        h2 = end_h
                        w1 = start_w
                        w2 = end_w
                    else:
                        if (end_h >= self.Height and end_w < self.Width):
                            h1 = self.Height - mask_height - 1
                            w1 = start_w
                            h2 = self.Height - 1
                            w2 = end_w
                        if (end_h < self.Height and end_w >= self.Width):
                            w1 = self.Width - mask_width - 1
                            h1 = start_h
                            w2 = self.Width - 1
                            h2 = end_h
                        if (end_h >= self.Height and end_w >= self.Width):
                            h1 = self.Height - mask_height - 1
                            h2 = self.Height - 1
                            w1 = self.Width - mask_width - 1
                            w2 = self.Width - 1
                    gaus = self.Gaussain_Border(h1, w1, h2, w2, self.sigma)
                    hist = [0.0] * self.Patterns
                    for c in range(h1, h2):
                        for r in range(w1, w2):
                            pattern = Img[c, r]
                            for k in range(self.Patterns):
                                if pattern == patterns[k]:
                                    if self.isgradiented == 1:
                                        hist[k] += gradients[c, r]
                                    else:
                                        hist[k] += 1 * gaus[c - h1, r - w1]
                    Histogram[:, count] = np.mat(hist).flatten().T
                    count += 1
        else:
            Histogram = np.mat(np.zeros((self.Patterns, self.w_num * self.h_num)))
            # print(shape(Histogram))
            count = 0
            for i in range(self.h_num):
                for j in range(self.w_num):
                    hist = [0.0] * self.Patterns
                    h1 = i * mask_height
                    h2 = (i + 1) * mask_height
                    w1 = j * mask_width
                    w2 = (j + 1) * mask_width
                    # gaus = self.Gaussain2D(h1, w1, h2, w2, self.sigma)
                    for c in range(h1, h2):
                        for r in range(w1, w2):
                            pattern = Img[c, r]
                            for k in range(self.Patterns):
                                if pattern == patterns[k]:
                                    if self.isgradiented == 1:
                                        hist[k] += gradients[c, r]
                                    else:
                                        # hist[k] += 1 * gaus[c - h1, r - w1]
                                        hist[k] += 1
                    Histogram[:, count] = np.mat(hist).flatten().T
                    count += 1
        return Histogram.flatten()

    # Use Gaussian to gain weights for each pixel, the weight is depend on its location
    def Gaussain2D(self, h1, w1, h2, w2, sigma):
        x0 = int((w1 + w2) / 2)
        y0 = int((h1 + h2) / 2)
        x = np.linspace(w1, w2 - 1, (w2 - w1))
        y = np.linspace(h1, h2 - 1, (h2 - h1))
        x, y = np.meshgrid(x, y)
        gaus = np.exp(-(((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2))))
        return gaus

    def Gaussain_Border(self, h1, w1, h2, w2, sigma):
        xc = int((w1 + w2) / 2)
        yc = int((h1 + h2) / 2)
        x = np.linspace(w1, xc - 1, (xc - w1))
        y = np.linspace(h1, yc - 1, (yc - h1))
        x, y = np.meshgrid(x, y)
        gaus_1 = np.exp(-(((x - w1) ** 2 + (y - h1) ** 2) / (2 * (sigma ** 2))))
        x = np.linspace(xc, w2 - 1, (w2 - xc))
        y = np.linspace(h1, yc - 1, (yc - h1))
        x, y = np.meshgrid(x, y)
        gaus_2 = np.exp(-(((x - w2 + 1) ** 2 + (y - h1) ** 2) / (2 * (sigma ** 2))))
        x = np.linspace(w1, xc - 1, (xc - w1))
        y = np.linspace(yc, h2 - 1, (h2 - yc))
        x, y = np.meshgrid(x, y)
        gaus_3 = np.exp(-(((x - w1) ** 2 + (y - h2 + 1) ** 2) / (2 * (sigma ** 2))))
        x = np.linspace(xc, w2 - 1, (w2 - xc))
        y = np.linspace(yc, h2 - 1, (h2 - yc))
        x, y = np.meshgrid(x, y)
        gaus_4 = np.exp(-(((x - w2 + 1) ** 2 + (y - h2 + 1) ** 2) / (2 * (sigma ** 2))))
        t1 = np.concatenate((gaus_1, gaus_2), axis=1)
        t2 = np.concatenate((gaus_3, gaus_4), axis=1)
        gaus = 1 - np.concatenate((t1, t2))
        return gaus

    def zeroMean(self, dataMat):
        meanVal = np.mean(dataMat, axis=0)  # axis = 0, calculate the mean of each column
        newData = dataMat - meanVal
        return newData, meanVal

    # According to the percentage, calculate the 'n'
    def percentage_pca(self, eigVals, percentage):
        sortArray = np.sort(eigVals)
        sortArray = sortArray[-1::-1]  # Reverse, from the maximal to the minimal
        print(np.shape(sortArray))
        arraySum = sum(sortArray) * percentage
        tmpSum = 0
        num = 0
        for i in sortArray:
            tmpSum += i
            num += 1
            if tmpSum >= arraySum:
                return num

    def pca(self, dataMat, percentage=0.9999):
        newData, meanVal = self.zeroMean(dataMat)
        covMat = np.cov(newData, rowvar=0)  # covariance matrix, rowvar = 0, represents each row is a sample
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))
        # n = self.percentage_pca(eigVals, percentage)  # calculate 'n'
        n = 200
        eigValIndice = np.argsort(eigVals)
        n_eigValIndice = eigValIndice[-1:-(n + 1):-1]
        n_eigVect = eigVects[:, n_eigValIndice]  # obtain 'n' eigen vector
        lowDDataMat = newData * n_eigVect  # low dimensional data matrix
        reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # the reconstructed data matrix
        return lowDDataMat, reconMat

    def percentage_svd(self, sigma, percentage):
        arraySum = np.sum(sigma) * percentage
        num = 0
        tmpSum = 0
        for i in sigma:
            tmpSum += i
            num += 1
            if tmpSum >= arraySum:
                return num

    def svd(self, dataMat, percentage=0.5):
        U, sigma, V = np.linalg.svd(dataMat)
        n = self.percentage_svd(sigma,percentage)
        print(n)
        reconData = np.matrix(U[:, :n]) * np.diag(sigma[:n]) * np.matrix(V[:n, :])
        print(np.shape(reconData))

    @fn_timer
    def run_LBP(self, path, isgradient, exp_para, sigma):
        self.isgradiented = isgradient
        self.exp_para = exp_para
        self.sigma = sigma
        self.Image_Num = len([file for file in os.listdir(path)
                              if os.path.isfile(os.path.join(path, file))])

        # Load images
        self.ids, FaceMat, self.gradients = self.loadImages(self.Image_Num, path, 0, 0)
        # Calculate LBP opearters for all loaded images
        self.LBPoperator = self.LBP(FaceMat)
        # Calculate histograms
        self.Histograms = self.createHist(self.Image_Num)
        for i in range(self.Image_Num):
            Histogram = self.calHistogram(self.LBPoperator[i, :], self.gradients[i, :])
            self.Histograms[i, :] = Histogram
        # lowData, reconData = self.pca(self.Histograms)
        # self.lowHistograms = lowData
        reconData = self.svd(self.Histograms)

    def createHist(self, image_num):
        if self.overlap_ratio > 0:
            Histograms = np.mat(
                np.zeros((image_num, self.Patterns * self.region_w_num * self.region_h_num)))
        else:
            Histograms = np.mat(np.zeros((image_num, self.Patterns * self.w_num * self.h_num)))
        return Histograms

    # recogniseImg: the image needed to be matched
    def recogniseFace(self, recog_num, recog_images, recog_gradients):
        weights = self.weights
        recog_ids = [0 for i in range(recog_num)]
        recog_ope = self.LBP(recog_images)
        recog_hists = self.createHist(recog_num)
        for i in range(recog_num):
            hist = self.calHistogram(recog_ope[i, :], recog_gradients[i, :])
            recog_hists[i, :] = hist
        recog_low_hists, reconData = self.pca(recog_hists)
        for r in range(recog_num):
            minIndex = 0
            minVals = np.inf
            # recog_hist = recog_hists[r,:]
            recog_hist = recog_low_hists[r, :]
            # Find the most close one, the smallest difference
            for i in range(np.shape(self.LBPoperator)[0]):
                # Histogram = self.Histograms[i, :]
                Histogram = self.lowHistograms[i, :]
                # Utilize Chi square distance to find the most close match
                if (len(weights) == 0):
                    distance = abs(((np.array(Histogram - recog_hist) ** 2).sum()) / (
                        (np.array(Histogram + recog_hist)).sum()))
                else:
                    if (self.overlap_ratio == 0):
                        regions = self.h_num * self.w_num
                    else:
                        regions = self.region_h_num * self.region_w_num
                    Histogram = Histogram.reshape(self.Patterns, regions)
                    recog_hist = recog_hist.reshape(self.Patterns, regions)
                    distance = 0
                    for index in range(regions):
                        distance += (((np.array(Histogram[:, index] - recog_hist[:, index])) ** 2).sum() / (
                            np.array(Histogram[:, index] + recog_hist[:, index])).sum()) * weights[index]
                    distance = abs(distance)
                if distance < minVals:
                    minIndex = i
                    minVals = distance
            recog_ids[r] = self.ids[minIndex]
        return recog_ids

        # Calculate the recognition rate

    def calculate_Accuracy(self, mypath, shift_vertical, shift_horizonal):
        # mypath = '/cs/home/jf231/Dissertation/CS5099/Yale_images/Set_3/'
        # shift_vertical = self.shift_vertical
        recog_num = len([file for file in os.listdir(mypath)
                         if os.path.isfile(os.path.join(mypath, file))])
        recog_ids, recog_faces, recog_gradients = self.loadImages(recog_num, mypath, shift_vertical, shift_horizonal)
        ids = self.recogniseFace(recog_num, recog_faces, recog_gradients)
        count = 0
        for i in range(recog_num):
            if ids[i] == recog_ids[i]:
                count += 1
        return float(count) / recog_num
