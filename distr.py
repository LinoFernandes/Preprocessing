import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


Window = np.array([90, 180, 365])
for window in Window:
        if sys.platform == "darwin":
            path = '/Users/Lino/PycharmProjects/Preprocessing/' + str(window) + 'd_TT.csv'
            Data = pd.read_csv(path)
        elif sys.platform == "win32":
            path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\' + str(window) + 'd_TT.csv'
            Data = pd.read_csv(path)
        #sys.path.append(path)

        Snapshots = Data.values[:,0]

        UniqueSnapshots = set(Snapshots)
        Histogram = []
        for whatever in UniqueSnapshots:
            Histogram.append(len(Snapshots[Snapshots == whatever]))
        z=[i for i in range(0,len(Histogram)) if Histogram[i] == 17]
        print(z)
        bins=np.arange(0,20,1)
        plt.hist(Histogram,bins,align='left',rwidth=0.5)
        bins=np.arange(0,20,1)
        plt.xticks(bins[:-1])
        plt.xlabel('Número de pontos')
        plt.ylabel('Distribuição_' + str(window) + 'd')
        plt.savefig('foo' + str(window) + '.png')
        plt.clf()

        z = [i for i in range(0,len(Histogram)) if Histogram[i]>=3]
        print('# ' + str(window) + 'd: ' + str(len(UniqueSnapshots)))
        print(len(z))