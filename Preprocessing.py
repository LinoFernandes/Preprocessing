import numpy as np
import pandas as pd
import sys
import os
import datetime as dt

Window = np.array([90, 180, 365])
for window in Window:
    for seed in range(1,6):
        if sys.platform == 'win32':
            path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\PreProcessedFolds\\' + str(window) + 'd_FOLDS\\S' + str(seed)
        elif sys.platform == 'darwin':
            path = '/Users/Lino/PycharmProjects/Preprocessing/PreProcessedFolds/' + str(window) + 'd_FOLDS/S' + str(seed)
        sys.path.append(path)
        for fold in range(1,11):
            try:
                if sys.platform == 'win32':
                    TrainPD = pd.read_csv(path + '\\' + str(window) + 'd_FOLDS_train_' + str(fold) + '.csv')
                    TestPD = pd.read_csv(path + '\\' + str(window) + 'd_FOLDS_test_' + str(fold) + '.csv')
                elif sys.platform == 'darwin':
                    TrainPD = pd.read_csv(path + '/' + str(window) + 'd_FOLDS_train_' + str(fold) + '.csv')
                    TestPD = pd.read_csv(path + '/' + str(window) + 'd_FOLDS_test_' + str(fold) + '.csv')
            except FileNotFoundError:
                continue
            auxTrain = TrainPD.columns[pd.isnull(TrainPD).any()].tolist() #Columns with nans
            auxTest = TestPD.columns[pd.isnull(TestPD).any()].tolist() #Columns with nans
            from sklearn.preprocessing import Imputer
            imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
            TrainPD[auxTrain] = imputer.fit_transform(TrainPD[auxTrain])
            TestPD[auxTest] = imputer.fit_transform(TestPD[auxTest])

            auxTrain = np.zeros(shape=[len(TrainPD), 1])
            auxTest = np.zeros(shape=[len(TestPD), 1])
            CheckTrain = TrainPD['Evolution'] == 'Y'
            CheckTest = TestPD['Evolution'] == 'Y'
            auxTrain[CheckTrain,] = 1
            auxTest[CheckTest,] = 1
            TrainPD['Evolution'] = auxTrain
            TestPD['Evolution'] = auxTest
            #
            CatCols = [5, 7, 8]
            CatCols=TrainPD[TrainPD.columns.values[CatCols]]
            for col in CatCols:
                if col == 'El Escorial reviewed criteria':
                    Words = ['PMA', 'def', 'pos', 'pro', 'sus']
                elif col == 'Envolved segment - 1st symptoms':
                    Words = ['B', 'B,AR', 'LL', 'UL']
                else:
                    Words = ['hemi', 'hemi cruz', 'hemi para', 'para', 'para cruz']
                auxTrain = np.zeros(shape=[len(TrainPD), 1])
                auxTest = np.zeros(shape=[len(TestPD), 1])
                for word in Words:
                    CheckTrain = TrainPD[col] == word
                    CheckTest = TestPD[col] == word
                    auxTrain[CheckTrain,] = Words.index(word)
                    auxTest[CheckTest,] = Words.index(word)
                TrainPD[col] = auxTrain
                TestPD[col] = auxTest


                # dummiesTrain = pd.get_dummies(TrainPD[col], columns=Words, drop_first=True)
                # dummiesTest = pd.get_dummies(TestPD[col], columns=Words, drop_first=True)
                # i = 0
                # auxTrain = np.zeros(shape=[len(TrainPD), len(Words)])
                # auxTest = np.zeros(shape=[len(TestPD), len(Words)])
                # for word in Words:
                #     CheckTrain = TrainPD[col] == word
                #     CheckTest = TestPD[col] == word
                #     auxTrain[CheckTrain, i] = 1
                #     auxTest[CheckTest,i] = 1
                #     i += 1
                # ColLoc = TrainPD.columns.get_loc(col)
                # TrainPD = TrainPD.drop(col, 1)
                # TestPD = TestPD.drop(col,1)
                # i = 0
                # for dummy in range(0,len(Words)):
                #     TrainPD.insert(ColLoc + i, Words[dummy], auxTrain[:,dummy])
                #     TestPD.insert(ColLoc + i, Words[dummy], auxTest[:, dummy])
                #     i += 1

            # from sklearn.preprocessing import MinMaxScaler
            # SC = MinMaxScaler()
            # SCCol=[x for x in TrainPD.columns.values if x not in ['Name','Evolution']]
            # TrainPD[SCCol] = SC.fit_transform(TrainPD.ix[:, SCCol])
            # TestPD[SCCol] = SC.fit_transform(TestPD.ix[:, SCCol])

            if sys.platform == 'win32':
                directory = ('C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\PreProcessedFoldsFinal\\' + str(window) + 'd_FOLDS\\S' + str(seed) + '\\')
            elif sys.platform == 'darwin':
                directory = ('/Users/Lino/PycharmProjects/Preprocessing/PreProcessedFoldsFinal/' + str(window) + 'd_FOLDS/S' + str(seed) + '/')
            if not os.path.exists(directory):
                os.makedirs(directory)
            filenameTrain = str(window) + 'd_FOLDS_train_' + str(fold) + '.csv'
            filenameTest = str(window) + 'd_FOLDS_test_' + str(fold) + '.csv'
            os.path.join(directory, filenameTrain)
            TrainPD.to_csv(directory+filenameTrain, sep=',',index=False)
            os.path.join(directory, filenameTest)
            TestPD.to_csv(directory+filenameTest, sep=',',index=False)