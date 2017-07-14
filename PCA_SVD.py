
import numpy as np
import cv2
import time
from functools import wraps


class PCA_SVD(object):
    def __init__(self, img_num, dimension):
        self.Img_Num = img_num
        self.k = 4
        self.dimension = dimension

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
        meanVal = np.mean(dataMat, axis=1)  # axis = 1, calculate the mean of each row
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

    def pca(self, dataMat, percentage=0.99):
        dataMat = dataMat.reshape(self.dimension, self.Img_Num)
        # dataMat = dataMat.T # Each row is an image
        newData, meanVal = self.zeroMean(dataMat)
        covMat = np.cov(newData)
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))
        # n = self.percentage_pca(eigVals, percentage)  # calculate 'n'
        n = self.k
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
    def trainBases(self, train_matrix, train_ids):
        self.train_ids = train_ids
        self.train_bases = np.mat(np.zeros((self.dimension * self.k, len(self.train_ids)), dtype=np.complex))
        for i in range(len(train_ids)):
            self.train_bases[:, i] = self.pca(train_matrix[i, :])
            # each row in train_matrix is for one set (one person)
        # import pandas as pd
        # train = pd.DataFrame(self.train_bases)
        # train.to_csv('train_gradiented_weighted.csv', sep=',', index=False)

    @fn_timer
    def getAccuracy(self, test_matrix, test_ids, index):
        test_bases = np.mat(np.zeros((self.dimension * self.k, len(test_ids)), dtype=np.complex))
        for i in range(len(test_ids)):
            test_bases[:, i] = self.pca(test_matrix[i, :])
        import pandas as pd
        test = pd.DataFrame(test_bases)
        # if index == 0:
        #     test.to_csv('test1_gradiented_weighted.csv', sep=',', index=False)
        # else:
        #     test.to_csv('test2_gradiented_weighted.csv', sep=',', index=False)
        count = 0
        for i in range(np.shape(self.train_bases)[1]):
            maxIndex = 0
            maxVals = -1
            B1 = self.train_bases[:, i].reshape(self.dimension, self.k).T
            for j in range(np.shape(test_matrix)[0]):
                B2 = test_bases[:, j].reshape(self.dimension, self.k)
                M = B1 * B2
                sigma = self.svd(M)
                if sigma > maxVals:
                    maxIndex = j
                    maxVals = sigma
            if self.train_ids[i] == test_ids[maxIndex]:
                count += 1
        print('Matched Set Number: %d' % count)
        return float(count) / len(test_ids)
