import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if sys.platform == "darwin":
    ALSFRS = pd.read_csv('/Users/Lino/PycharmProjects/Preprocessing/alsfrs.csv', na_values=np.NaN)
    Demographics = pd.read_csv('/Users/Lino/PycharmProjects/Preprocessing/demo.csv', na_values=np.NaN)
elif sys.platform == "win32":
    ALSFRS = pd.read_csv('C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\alsfrs.csv')
    Demographics = pd.read_csv('C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\demo.csv')


Demographics = Demographics.values
ALSFRS = ALSFRS.values

strentry = [i for i in range(0, len(ALSFRS)) if ALSFRS[i,1] == 'n' or ALSFRS[i,1] == 'N']
ALSFRS[strentry,1] = np.NaN

Score = []
for entry in range(0,len(ALSFRS)):
    try:
        ALSFRS[entry, 1] = int(ALSFRS[entry, 1])
        score = (48 - ALSFRS[entry, 1]) / Demographics[entry,1]
        Score.append(score)
    except ValueError:
        Score.append(np.NaN)


