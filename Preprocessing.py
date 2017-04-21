import numpy as np
import pandas as pd
import sys
import os

Window = np.array([90, 180, 365])
for window in Window:
    for seed in range(1,6):
        path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\PreProcessedFolds\\' + str(window) + 'd_FOLDS\\S' + str(seed)
        sys.path.append(path)
        for fold in range(1,11):
            TrainPD = pd.read_csv(path + '\\' + str(window) + 'd_FOLDS_train_' + str(fold) + '.csv')
            TestPD = pd.read_csv(path + '\\' + str(window) + 'd_FOLDS_test_' + str(fold) + '.csv')
            Train = TrainPD.values
            Test = TestPD.values

            auxTrain = TrainPD.columns[pd.isnull(TrainPD).any()].tolist() #Columns with nans
            auxTest = TestPD.columns[pd.isnull(TestPD).any()].tolist() #Columns with nans
            from sklearn.preprocessing import Imputer
            imputer = Imputer(missing_values='NaN', strategy='mean',axis=0);
            TrainPD[auxTrain] = imputer.fit_transform(TrainPD[auxTrain])
            TestPD[auxTest] = imputer.fit_transform(TestPD[auxTest])

            from sklearn.preprocessing import LabelEncoder
            LE = LabelEncoder()
            TrainPD['Evolution'] = LE.fit_transform(TrainPD['Evolution'])
            TestPD['Evolution'] = LE.fit_transform(TestPD['Evolution'])

            CatCols = [5, 7, 8]
            for col in CatCols:
                dummies = pd.get_dummies(TrainPD[TrainPD.columns.values[col]])


            NewTrain = pd.DataFrame(Train, index = None, columns = TrainPD.columns.values)
            NewTest = pd.DataFrame(Test, index = None, columns = TestPD.columns.values)
            directory = ('C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\PreProcessedFolds\\' + str(window) + 'd_FOLDS\\S' + str(seed) + '\\')
            if not os.path.exists(directory):
                os.makedirs(directory)
            filenameTrain = str(window) + 'd_FOLDS_train_' + str(fold) + '.csv'
            filenameTest = str(window) + 'd_FOLDS_test_' + str(fold) + '.csv'
            os.path.join(directory, filenameTrain)
            NewTrain.to_csv(directory+filenameTrain, sep=',')
            os.path.join(directory, filenameTest)
            NewTest.to_csv(directory+filenameTest, sep=',')