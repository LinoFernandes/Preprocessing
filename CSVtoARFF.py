import numpy as np
import pandas as pd
import sys
import os.path
import datetime as dt

Window = np.array([90, 180, 365])
for window in Window:
    for seed in range(1,6):
        if sys.platform == 'win32':
            path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing2\\PreProcessedFolds\\' + str(window) + 'd_FOLDS\\S' + str(seed)
        elif sys.platform == 'darwin':
            path = '/Users/Lino/PycharmProjects/Preprocessing2/PreProcessedFolds/' + str(window) + 'd_FOLDS/S' + str(seed)
        sys.path.append(path)
        for fold in range(1,11):
            try:
                if sys.platform == 'win32':
                    TrainPD = pd.read_csv(path + '\\' + str(window) + 'd_FOLDS_train_' + str(fold) + '.csv')
                    TestPD = pd.read_csv(path + '\\' + str(window) + 'd_FOLDS_test_' + str(fold) + '.csv')
                elif sys.platform == 'darwin':
                    TrainPD = pd.read_csv(path + '/' + str(window) + 'd_FOLDS_train_' + str(fold) + '.csv')
                    TestPD = pd.read_csv(path + '/' + str(window) + 'd_FOLDS_test_' + str(fold) + '.csv')
            except OSError:
                continue

            if sys.platform == 'win32':
                directory = ('C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\PreProcessedFoldsARFF\\' + str(window) + 'd_FOLDS\\S' + str(seed) + '\\')
            elif sys.platform == 'darwin':
                directory = ('/Users/Lino/PycharmProjects/Preprocessing2/PreProcessedFoldsARFF/' + str(window) + 'd_FOLDS/S' + str(seed) + '/')
            if not os.path.exists(directory):
                os.makedirs(directory)
            completeName = os.path.join(directory, str(window) + 'd_FOLDS_train_' + str(fold) + '.arff')
            fTrain = open(completeName, 'w')
            completeName = os.path.join(directory, str(window) + 'd_FOLDS_test_' + str(fold) + '.arff')
            fTest = open(completeName, 'w')

            fTrain.write('@relation ' + str(window) + 'd_FOLDS_train_' + str(fold) + '\n\n')
            fTest.write('@relation ' + str(window) + 'd_FOLDS_test_' + str(fold) + '\n\n')

            for Column in range(0,len(TrainPD.columns.values)):
                CatCols = [5, 7, 8]
                if Column in CatCols:
                    checkColTrain = set(TrainPD[TrainPD.columns.values[Column]])
                    checkColTest = set(TestPD[TestPD.columns.values[Column]])
                    fTrain.write('@attribute ' + TrainPD.columns.values[Column] + ' {')
                    fTest.write('@attribute ' + TrainPD.columns.values[Column] + ' {')
                    for i in range(0,len(list(checkColTrain))):
                        fTrain.write(list(checkColTrain)[i] + ',')
                        if i == len(list(checkColTrain))-1:
                            fTrain.write(list(checkColTrain)[i] + '}\n')
                    for i in range(0, len(list(checkColTest))):
                        fTest.write(list(checkColTest)[i] + ',')
                        if i == len(list(checkColTest)) - 1:
                            fTest.write(list(checkColTest)[i] + '}\n')
                elif '%' in TrainPD.columns.values[Column]:
                    aux = [i for i in range(0,len(TrainPD.columns.values[Column])) if TrainPD.columns.values[Column][i] == '%']
                    newCol = ""
                    for i in range(0,len(TrainPD.columns.values[Column])):
                        if i == aux[0]:
                            newCol += "\\%"
                        else:
                            newCol += str(TrainPD.columns.values[Column][i])
                    fTrain.write('@attribute \'' + newCol + '\' numeric\n')
                    fTest.write('@attribute \'' + newCol + '\' numeric\n')
                elif TrainPD.columns.values[Column] == 'Evolution':
                    fTrain.write('@attribute Evolution {Y,N}')
                    fTest.write('@attribute Evolution {Y,N}')
                else:
                    fTrain.write('@attribute ' + TrainPD.columns.values[Column] + ' numeric\n')
                    fTest.write('@attribute ' + TrainPD.columns.values[Column] + ' numeric\n')
            fTrain.write('\n@data\n')
            for row in range(0,len(TrainPD)):
                TrainAux = TrainPD.values
                isnotstr = [i for i in range(0,len(TrainAux[row][:])) if type(TrainAux[row][i]) != type('')]
                isnan = []
                aux = TrainAux[row][:].tolist()
                for i in isnotstr:
                    if np.isnan(TrainAux[row][i]):
                        aux[i] = '?'
                for item in aux:
                    if item != '?':
                        fTrain.write(str(item) + ',')
                    else:
                        fTrain.write(item[0] + ',')
