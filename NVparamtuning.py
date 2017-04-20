import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import sys
Window = np.array([90, 180, 365])

def categoricalData(data):
    from scipy.stats import mode
    Counts = mode(data)
    return Counts.mode[0]

for window in Window:
    for seed in range(1,2):
        path = '/Users/Lino/PycharmProjects/Preprocessing/FOLDS/' + str(window) + 'd_FOLDS/S' + str(seed)
        sys.path.append(path)
        for fold in range(1,2):
            #print(path + '/' + str(window) + 'd_FOLDS_train_' + str(fold))
            Train = pd.read_csv(path + '/' + str(window) + 'd_FOLDS_train_' + str(fold) + '.csv')
            Test = pd.read_csv(path + '/' + str(window) + 'd_FOLDS_test_' + str(fold) + '.csv')
            X = Train.iloc[:,:-2].values
            Y = Train.iloc[:,np.shape(Train)[1]-2].values
            #print(np.shape(X))
            print(X)
            #model = GaussianNB()
            #pred = model.partial_fit()


