import numpy as np
import pandas as pd
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics


Window = np.array([90, 180, 365])
roc = []
spec = []
sens = []
prec = []
for window in Window:
    for seed in range(1,6):
        if sys.platform == "darwin":
            path = '/Users/Lino/PycharmProjects/Preprocessing/PreProcessedFoldsFinal/' + str(window) + 'd_FOLDS/S' + str(
                seed)
        elif sys.platform == "win32":
            path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\PreProcessedFoldsFinal\\' + str(
                window) + 'd_FOLDS\\S' + str(seed)
        sys.path.append(path)
        for fold in range(1,11):
            try:
                if sys.platform == "darwin":
                    TrainPD = pd.read_csv(path + '/' + str(window) + 'd_FOLDS_train_' + str(fold) + '.csv')
                    TestPD = pd.read_csv(path + '/' + str(window) + 'd_FOLDS_test_' + str(fold) + '.csv')
                    directory = ('/Users/Lino/PycharmProjects/Preprocessing/NV/' + str(window) + 'd_FOLDS/S' + str(seed) + '/')
                elif sys.platform == "win32":
                    TrainPD = pd.read_csv(path + '\\' + str(window) + 'd_FOLDS_train_' + str(fold) + '.csv')
                    TestPD = pd.read_csv(path + '\\' + str(window) + 'd_FOLDS_test_' + str(fold) + '.csv')
                    directory = ('C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NV\\' + str(window) + 'd_FOLDS\\S' + str(seed) + '\\')
            except FileNotFoundError:
                continue

            svc = SVC(C=1,kernel='poly',degree=2)
            NB = GaussianNB()
            RF = RandomForestClassifier()
            if window == 90:
                SelectedFeatures = ['Gender','BMI','Age at onset','El Escorial reviewed criteria','Onset form','ALS-FRS-R','R','SpO2<90%','Pattern','%FVC','%MEP','PO2','PhrenMeanAmpl','PhrenMeanArea']
            elif window == 180:
                SelectedFeatures = ['Name','Gender','BMI','Age at onset','Envolved segment - 1st symptoms','ALS-FRS-R','R','SpO2<90%','Dips4%','Pattern','%FVC','%MEP','PO2','PhrenMeanAmpl','PhrenMeanArea']
            else:
                SelectedFeatures = ['Name','Gender','BMI','Age at onset','Envolved segment - 1st symptoms','1st symptoms - 1st visit','ALS-FRS','ALS-FRS-R',
                                    'R','SpO2<90%','Dips/h<4%','Dips3%','Pattern','%FVC','%MEP','PO2','peso','PhrenMeanAmpl','PhrenMeanArea']
            NB.fit(TrainPD.ix[:, TrainPD.columns != 'Evolution'], TrainPD.ix[:, 'Evolution'])
            #svc.fit(TrainPD.ix[:, SelectedFeatures], TrainPD.ix[:, 'Evolution'])
            EvoPred = NB.predict(TestPD.ix[:, TrainPD.columns != 'Evolution'])
            #EvoPred = svc.predict(TestPD.ix[:, SelectedFeatures])
            CM = metrics.confusion_matrix(y_true=TestPD.ix[:, 'Evolution'],y_pred=EvoPred)
            roc.append(metrics.roc_auc_score(TestPD.ix[:, 'Evolution'],EvoPred))
            spec.append(CM[0][0]/(CM[0][0]+CM[1][0]))
            sens.append(metrics.recall_score(TestPD.ix[:, 'Evolution'],EvoPred))
            prec.append(metrics.precision_score(TestPD.ix[:, 'Evolution'],EvoPred))

    print(str(window) + 'd || AUC:' + str(np.mean(roc)) + ' || Spec: ' + str(np.mean(spec)) + ' || Sens: ' + str(np.mean(sens)) + ' || Prec: ' + str(np.mean(prec)))