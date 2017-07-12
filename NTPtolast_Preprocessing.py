import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

for NTP in range(2,7):
    Begins = np.array(range(0,NTP))
    Window = np.array([90, 180, 365])
    for window in Window:
        for begin in Begins:
            if sys.platform == "darwin":
                path = '/Users/Lino/PycharmProjects/Preprocessing/NTP/' + str(NTP) + 'TP/' + str(window) + 'd_'+ str(NTP) + '.csv'
                Data = pd.read_csv(path)
                directory = '/Users/Lino/PycharmProjects/Preprocessing/NTPtoLast/' + str(NTP) + 'TP/'
            elif sys.platform == "win32":
                path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\' + str(window) + 'd_TT.csv'
                Data = pd.read_csv(path)
                directory = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NTPtoLast\\' + str(NTP) + 'TP\\'
            #sys.path.append(path)

            if begin == 0:
                filenameTrain = str(window) + 'd_' + str(begin) + 'to' + str(NTP-1) + '.csv'
                #filenameTest = str(window) + 'd_FOLDS_test_' + str(fold) + '.csv'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                os.path.join(directory, filenameTrain)
                Data.to_csv(directory + filenameTrain, sep=',', index=False)
                #os.path.join(directory, filenameTest)
                #TestPD.to_csv(directory + filenameTest, sep=',', index=False)
                continue

            Columns = Data.columns.values

            ToBeDeleted = np.array(range(10,33 + (23*(begin-1))))
            for col in Data.columns.values[ToBeDeleted]:
                Data = Data.drop(col,axis=1)
            if not os.path.exists(directory):
                os.makedirs(directory)
            filenameTrain = str(window) + 'd_' + str(begin) + 'to' + str(NTP-1) + '.csv'
            #filenameTest = str(window) + 'd_FOLDS_test_' + str(fold) + '.csv'
            os.path.join(directory, filenameTrain)
            Data.to_csv(directory + filenameTrain, sep=',', index=False)
            #os.path.join(directory, filenameTest)
            #TestPD.to_csv(directory + filenameTest, sep=',', index=False)



