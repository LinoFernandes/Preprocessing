import numpy as np
import pandas as pd
import sys
import os

Window = np.array([90, 180, 365])
def categoricalData(data):
    from collections import Counter
    Counts = Counter(data)
    if isinstance(Counts.most_common(1)[0][0],str) == False:
        return Counts.most_common(2)[1][0]
    else:
        return Counts.most_common(1)[0][0]


for window in Window:
    for seed in range(1,6):
        path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\FOLDS\\' + str(window) + 'd_FOLDS\\S' + str(seed)
        sys.path.append(path)
        for fold in range(1,11):
            #print(path + '/' + str(window) + 'd_FOLDS_train_' + str(fold))
            TrainPD = pd.read_csv(path + '\\' + str(window) + 'd_FOLDS_train_' + str(fold) + '.csv')
            TestPD = pd.read_csv(path + '\\' + str(window) + 'd_FOLDS_test_' + str(fold) + '.csv')
            Train = TrainPD.values
            Test = TestPD.values
            CatCol = [5,7,8]
            for col in CatCol:
                ModeTrain = categoricalData(Train[:,col])
                ModeTest = categoricalData(Test[:,col])
                NaNindexTrain= [i for i in range(len(Train[:,col])) if type(Train[:,col][i]) != type(str())]
                NaNindexTest = [i for i in range(len(Test[:, col])) if type(Test[:, col][i]) != type(str())]
                Train[NaNindexTrain, col] = ModeTrain
                Test[NaNindexTest, col] = ModeTest
            NewTrain = pd.DataFrame(Train, index = None, columns = TrainPD.columns.values)
            NewTest = pd.DataFrame(Test, index = None, columns = TestPD.columns.values)
            directory = ('C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\PreProcessedFolds\\' + str(window) + 'd_FOLDS\\S' + str(seed) + '\\')
            if not os.path.exists(directory):
                os.makedirs(directory)
            filenameTrain = str(window) + 'd_FOLDS_train_' + str(fold) + '.csv'
            filenameTest = str(window) + 'd_FOLDS_test_' + str(fold) + '.csv'
            os.path.join(directory, filenameTrain)
            NewTrain.to_csv(directory+filenameTrain, sep=',',index=False)
            os.path.join(directory, filenameTest)
            NewTest.to_csv(directory+filenameTest, sep=',',index=False
                           )