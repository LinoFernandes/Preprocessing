import pandas as pd
import numpy as np
import sys
import os

Window = np.array([90, 180, 365])

if sys.platform == "darwin":
    completeName = os.path.join('/Users/Lino/Desktop','Evo.txt')
elif sys.platform == "win32":
    completeName = os.path.join('C:\\Users\\Lino\\Desktop','Evo.txt')
for window in Window:
    if sys.platform == "darwin":
        completeName = os.path.join('/Users/Lino/Desktop', 'Evo_' + str(Window) + 'TP.csv')
        Perf = open(completeName, 'a')
        Perf.write('k,#TP,#Patients,Evo(Y),Evo(N)\n')
    elif sys.platform == "win32":
        path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NTP\\' + str(ntp) + 'TP'
    for ntp in range(1,9):
        if sys.platform == "darwin":
            path = '/Users/Lino/PycharmProjects/Preprocessing/NTP/' + str(ntp) + 'TP'
        sys.path.append(path)

        Data = pd.read_csv(path + '/' + str(window) + 'd_' + str(ntp) + '.csv')

        Evo = Data.iloc[:,len(Data.values[0])-1].values
        NumPatients = len(Data.iloc[:,0])

        Y = len(np.where(Evo == 'Y')[0])
        N = len(Evo) - Y

        PY = (Y/len(Evo))*100
        PN = (N/len(Evo))*100

        Perf.write(str(window) + ',' + str(ntp) + ','  + str(NumPatients) + ','  + str(Y) + '('+ str(round(PY,4)) +')' + ','  + str(N) + '('+ str(round(PN,4)) +')\n')
        #Perf.write('Y:' + str(Y) + '\n')
        #Perf.write('N:' + str(N) + '\n')
        #Perf.write('PY:' + str(PY) + '\n')
        #Perf.write('PN:' + str(PN) + '\n')
        #Perf.write('#Pat:' + str(NumPatients) + '\n')







