import numpy as np
import pandas as pd
import sys
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

Window = np.array([90, 180, 365])
for window in Window:
    for seed in range(1,6):
        path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\PreProcessedFoldsFinal\\' + str(window) + 'd_FOLDS\\S' + str(seed)
        sys.path.append(path)
        for fold in range(1,11):
            try:
                TrainPD = pd.read_csv(path + '\\' + str(window) + 'd_FOLDS_train_' + str(fold) + '.csv')
            except FileNotFoundError:
                continue
            degree = np.array([1,2,3])
            model = SVC(kernel='poly')
            grid = GridSearchCV(estimator=model, param_grid=dict(degree = degree))
            grid.fit(TrainPD.ix[:, TrainPD.columns != 'Evolution'], TrainPD.ix[:, 'Evolution'] )
            print(grid)
            print(grid.best_score_)
            print(grid.best_estimator_.degree)