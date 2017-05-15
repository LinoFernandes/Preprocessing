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
Score = np.array(Score)
NotNaN = [i for i in range(0, len(Score)) if not np.isnan(Score[i]) or Score[i] > 0]

hist, bins = np.histogram(Score[NotNaN], bins=1000)
freq = 5
hist[np.where(hist <= freq)] = 0
z=np.percentile(Score[NotNaN], np.arange(0, 100, 25))

# Plot
bins = bins[hist != 0]
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
print(len(bins))
print(len(hist[hist!=0]))
print(center)
plt.bar(bins, hist[hist!=0], align='center', width=width)
for line in z:
    plt.axvline(x=line)
plt.savefig('TimeEvo.png')