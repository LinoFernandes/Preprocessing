import numpy as np
import pandas as pd
import sys
import os
import datetime as dt

Window = np.array([90, 180, 365])
for window in Window:
    for seed in range(1,6):
        path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\PreProcessedFolds\\' + str(window) + 'd_FOLDS\\S' + str(seed)
        sys.path.append(path)
        for fold in range(1,11):
            try:
                TrainPD = pd.read_csv(path + '\\' + str(window) + 'd_FOLDS_train_' + str(fold) + '.csv')
                TestPD = pd.read_csv(path + '\\' + str(window) + 'd_FOLDS_test_' + str(fold) + '.csv')
            except FileNotFoundError:
                continue
            auxTrain = TrainPD.columns[pd.isnull(TrainPD).any()].tolist() #Columns with nans
            auxTest = TestPD.columns[pd.isnull(TestPD).any()].tolist() #Columns with nans
            from sklearn.preprocessing import Imputer
            imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
            TrainPD[auxTrain] = imputer.fit_transform(TrainPD[auxTrain])
            TestPD[auxTest] = imputer.fit_transform(TestPD[auxTest])

            from sklearn.preprocessing import LabelEncoder
            LE = LabelEncoder()
            TrainPD['Evolution'] = LE.fit_transform(TrainPD['Evolution'])
            TestPD['Evolution'] = LE.fit_transform(TestPD['Evolution'])

            CatCols = [5, 7, 8]
            CatCols=TrainPD[TrainPD.columns.values[CatCols]]
            for col in CatCols:
                if col == 'El Escorial reviewed criteria':
                    Words = ['PLS', 'PMA', 'def', 'pos', 'pro', 'sus']
                elif col == 'Envolved segment - 1st symptoms':
                    Words = ['AR', 'B', 'B,AR', 'LL', 'UL']
                else:
                    Words = ['CROSSED', 'hemi', 'hemi cruz', 'hemi para', 'para', 'para cruz']
                dummiesTrain = pd.get_dummies(TrainPD[col], prefix=Words, drop_first=True)
                dummiesTest = pd.get_dummies(TestPD[col], prefix=Words, drop_first=True)
                aux = TrainPD.columns.get_loc(col)
                TrainPD = TrainPD.drop(col, 1)
                TestPD = TestPD.drop(col,1)
                i = 0
                for dummy in dummiesTrain.columns.values:
                    TrainPD.insert(aux + i, dummy, dummiesTrain[dummy])
                    i += 1
                i = 0
                for dummy in dummiesTest.columns.values:
                    TestPD.insert(aux + i, dummy, dummiesTest[dummy])
                    i += 1

            directory = ('C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\PreProcessedFoldsFinal\\' + str(window) + 'd_FOLDS\\S' + str(seed) + '\\')
            if not os.path.exists(directory):
                os.makedirs(directory)
            filenameTrain = str(window) + 'd_FOLDS_train_' + str(fold) + '.csv'
            filenameTest = str(window) + 'd_FOLDS_test_' + str(fold) + '.csv'
            os.path.join(directory, filenameTrain)
            TrainPD.to_csv(directory+filenameTrain, sep=',',index=False)
            os.path.join(directory, filenameTest)
            TestPD.to_csv(directory+filenameTest, sep=',',index=False)