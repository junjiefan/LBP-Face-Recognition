import os
import math
import numpy as np
import cv2
import time
from functools import wraps


class PCA_SVD(object):
    def __init__(self, img_num):
        self.Img_Num = img_num

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

    def zeroMean(self, dataMat):
        meanVal = np.mean(dataMat, axis=1)  # axis = 0, calculate the mean of each column
        newData = dataMat - meanVal
        return newData, meanVal

    # According to the percentage, calculate the 'n'
    def percentage_pca(self, eigVals, percentage):
        sortArray = np.sort(eigVals)
        sortArray = sortArray[-1::-1]  # Reverse, from the maximal to the minimal
        arraySum = sum(sortArray) * percentage
        tmpSum = 0
        num = 0
        for i in sortArray:
            tmpSum += i
            num += 1
            if tmpSum >= arraySum:
                return num

    def pca(self, dataMat, percentage=0.9999):
        dataMat = dataMat.reshape(3776, self.Img_Num)
        # dataMat = dataMat.T # Each row is an image
        newData, meanVal = self.zeroMean(dataMat)
        covMat = np.cov(newData)
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))
        # n = self.percentage_pca(eigVals, percentage)  # calculate 'n'
        n = 5
        eigValIndice = np.argsort(eigVals)
        n_eigValIndice = eigValIndice[-1:-(n + 1):-1]
        n_eigVect = eigVects[:, n_eigValIndice]  # obtain 'n' eigen vector
        # lowDDataMat = n_eigVect.T * newData  # low dimensional data matrix
        return n_eigVect.flatten().T

    def percentage_svd(self, sigma, percentage):
        arraySum = np.sum(sigma) * percentage
        num = 0
        tmpSum = 0
        for i in sigma:
            tmpSum += i
            num += 1
            if tmpSum >= arraySum:
                print(tmpSum)
                return num

    def svd(self, dataMat, percentage=0.5):
        U, sigma, V = np.linalg.svd(dataMat)
        return sigma[0]
        # n = self.percentage_svd(sigma, percentage)
        # reconData = np.matrix(U[:, :n]) * np.diag(sigma[:n]) * np.matrix(V[:n, :])
        # print(np.shape(reconData))

    @fn_timer
    def run_PCA(self, train_matrix, train_ids, test_matrix, test_ids):
        train_bases = np.mat(np.zeros((3776 * 5, len(train_ids)), dtype=np.complex))
        test_bases = np.mat(np.zeros((3776 * 5, len(test_ids)), dtype=np.complex))
        for i in range(len(train_ids)):
            train_bases[:, i] = self.pca(train_matrix[i, :])
        for i in range(len(test_ids)):
            test_bases[:, i] = self.pca(test_matrix[i, :])
        count = 0
        for i in range(np.shape(train_matrix)[0]):
            maxIndex = 0
            maxVals = -1
            B1 = train_bases[:, i].reshape(3776, 5).T
            for j in range(np.shape(test_matrix)[0]):
                B2 = test_bases[:, j].reshape(3776, 5)
                M = B1 * B2
                print('Shape of M:')
                print(np.shape(M))
                sigma = self.svd(M)
                if sigma > maxVals:
                    maxIndex = j
                    maxVals = sigma
            if train_ids[i] == test_ids[maxIndex]:
                count += 1
        print('count: %d' % count)
        return float(count) / len(test_ids)
