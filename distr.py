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
        print(len([i for i in range(0,len(Histogram)) if Histogram[i]>3]))
        plt.hist(Histogram)
        plt.savefig('foo' + str(window) + '.png')
        plt.gcf().clear()