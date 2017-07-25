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
                Quartiles = pd.read_csv('/Users/Lino/PycharmProjects/Classification/Quartiles.csv')
                Scores = pd.read_csv('/Users/Lino/PycharmProjects/Classification/Scores.csv')
                #directory = '/Users/Lino/PycharmProjects/Preprocessing/' + str(window) + 'd_TT.csv'
        elif sys.platform == "win32":
                path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\' + str(window) + 'd_TT.csv'
                Data = pd.read_csv(path)
                #directory = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\' + str(NTP) + 'TP\\'

        Quartiles = np.array(Quartiles.values)
        Scores = np.array(Scores.values)
        Data = Data.drop(['firstDate', 'lastDate', 'medianDate', 'Days-Until-NIV'], 1)


        Slow = [i for i in range(0,len(Scores)) if Scores[i] <= Quartiles[1]]
        Fast = [i for i in range(0,len(Scores)) if Scores[i] >= Quartiles[len(Quartiles)-1]]

        DataIndex = []
        for index in Slow:
            DataIndex.append([i for i in range(0,len(Data.values)) if Data.values[i,0] == index])
        DataIndex = np.array(DataIndex)
        Slow = pd.DataFrame(Data.values[DataIndex], index=None, columns=Data.columns.values)

        DataIndex = []
        for index in Fast:
            DataIndex.append([i for i in range(0,len(Data.values)) if Data.values[i,0] == index])
        DataIndex = np.array(DataIndex)
        Fast = pd.DataFrame(Data.values[Fast], index=None, columns=Data.columns.values)

        Fast.to_csv('/Users/Lino/PycharmProjects/Preprocessing/'+ str(window) + 'd_TT_Fast.csv', index=False)
        Slow.to_csv('/Users/Lino/PycharmProjects/Preprocessing/' + str(window) + 'd_TT_Slow.csv',index=False)

