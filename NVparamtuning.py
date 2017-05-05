import numpy as np
import pandas as pd
import sys
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import os
import datetime as dt

Window = np.array([90, 180, 365])
for window in Window:
    for seed in range(1,6):
        if sys.platform == "darwin":
            path = '/Users/Lino/PycharmProjects/Preprocessing/PreProcessedFoldsFinal/' + str(window) + 'd_FOLDS/S' + str(
                seed)
        elif sys.platform == "win32":
            path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\PreProcessedFoldsFinal\\' + str(
                window) + 'd_FOLDS\\S' + str(seed)
        sys.path.append(path)
        for fold in range(1,11):
            try:
                if sys.platform == "darwin":
                    TrainPD = pd.read_csv(path + '/' + str(window) + 'd_FOLDS_train_' + str(fold) + '.csv')
                    TestPD = pd.read_csv(path + '/' + str(window) + 'd_FOLDS_test_' + str(fold) + '.csv')
                    directory = ('/Users/Lino/PycharmProjects/Preprocessing/NV/' + str(window) + 'd_FOLDS/S' + str(seed) + '/')
                elif sys.platform == "win32":
                    TrainPD = pd.read_csv(path + '\\' + str(window) + 'd_FOLDS_train_' + str(fold) + '.csv')
                    TestPD = pd.read_csv(path + '\\' + str(window) + 'd_FOLDS_test_' + str(fold) + '.csv')
                    directory = ('C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NV\\' + str(window) + 'd_FOLDS\\S' + str(seed) + '\\')
            except FileNotFoundError:
                continue

            # Create the RFE object and rank each pixel
            NB = GaussianNB()
            svc = SVC(C = 1, kernel = 'linear')
            rfe = RFE(estimator=svc)
            rfe.fit(TrainPD.ix[:, TrainPD.columns != 'Evolution'], TrainPD.ix[:, 'Evolution'])
            ranking = rfe.ranking_()
            print(ranking)