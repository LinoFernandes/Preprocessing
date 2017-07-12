import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

Window = np.array([90, 180, 365])

for NTP in range(1,9):
    for window in Window:
            if sys.platform == "darwin":
                path = '/Users/Lino/PycharmProjects/Preprocessing/' + str(window) + 'd_TT.csv'
                Data = pd.read_csv(path)
                directory = '/Users/Lino/PycharmProjects/Preprocessing/NTP/' + str(NTP) + 'TP/'
            elif sys.platform == "win32":
                path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\' + str(window) + 'd_TT.csv'
                Data = pd.read_csv(path)
                directory = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NTP\\' + str(NTP) + 'TP\\'
            #sys.path.append(path)
            Data = Data.drop(['firstDate', 'lastDate', 'medianDate', 'Days-Until-NIV'], 1)
            Snapshots = Data.values[:, 0]
            UniqueSnapshots = set(Snapshots)
            Set = []
            for whatever in UniqueSnapshots:
                print(Snapshots[Snapshots == whatever])
                NTPs = Snapshots[Snapshots == whatever]
                if len(NTPs) >= NTP:
                    Set.append(NTPs[0])
            Ixs = [i for i in range(0, len(Snapshots)) if Snapshots[i] in Set]

            NTPData = np.array(Data.ix[Ixs,:])
            TPs = []
            for timepoint in Set:
                auxTPs = []
                ix = np.where(NTPData[:,0] == timepoint)[0]
                auxTPs.append(NTPData[ix[0],0:33])
                auxTPs.append(NTPData[ix[1:NTP],10:33])
                auxTPs.append(NTPData[ix[NTP-1],33])
                #print(auxTPs[2])
                TPs.append(np.append(np.append(auxTPs[0],auxTPs[1][:]),auxTPs[2]))
                #print(TPs)
                print(len(TPs[0]))

            ColumnsTrain = list(Data.columns.values)
            ColumnsTrain[10:34] = [s + '_0' for s in ColumnsTrain[10:34]]
            del ColumnsTrain[33]
            for i in range(0,NTP-1):
                for col in range(10,np.shape(Data)[1]-1):
                    ColumnsTrain.append(Data.columns.values[col] + '_' + str(i+1))
            ColumnsTrain.append(Data.columns.values[33] + '_' + str(NTP-1))
            print(ColumnsTrain)
            if not os.path.exists(directory):
                os.makedirs(directory)
            filenameTrain = str(window) + 'd_' + str(NTP) + '.csv'
            #filenameTest = str(window) + 'd_FOLDS_test_' + str(fold) + '.csv'
            os.path.join(directory, filenameTrain)
            TP = pd.DataFrame(TPs, index=None, columns=ColumnsTrain)
            TP.to_csv(directory + filenameTrain, sep=',', index=False)
            #os.path.join(directory, filenameTest)
            #TestPD.to_csv(directory + filenameTest, sep=',', index=False)



