import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats
import os



if sys.platform == "darwin":
    ALSFRS = pd.read_csv('/Users/Lino/PycharmProjects/Preprocessing2/alsfrs.csv', na_values=np.NaN)
    Demographics = pd.read_csv('/Users/Lino/PycharmProjects/Preprocessing2/demo.csv', na_values=np.NaN)
elif sys.platform == "win32":
    ALSFRS = pd.read_csv('C:\\Users\\Lino\\PycharmProjects\\Preprocessing2\\alsfrs.csv')
    Demographics = pd.read_csv('C:\\Users\\Lino\\PycharmProjects\\Preprocessing2\\demo.csv')


Demographics = Demographics.values
ALSFRS = ALSFRS.values

strentry = [i for i in range(0, len(ALSFRS)) if ALSFRS[i,1] == 'n' or ALSFRS[i,1] == 'N']
ALSFRS[strentry,1] = np.NaN

Score = []
for entry in range(0,len(ALSFRS)):
    try:
        ALSFRS[entry, 1] = int(ALSFRS[entry, 1])
        score = (40 - ALSFRS[entry, 1]) / Demographics[entry,1]
        Score.append(score)
    except ValueError:
        Score.append(np.NaN)
Score = np.array(Score)
NotNaN = [i for i in range(0, len(Score)) if not np.isnan(Score[i]) or Score[i] > 0]
print(len(Score[NotNaN]))
hist, bins = np.histogram(Score[NotNaN], bins=500)
freq = 5
hist[np.where(hist <= freq)] = 0
z1=Score[NotNaN]
z=np.percentile(z1, np.arange(0, 100, 25))
z=np.percentile(Score[NotNaN], np.arange(0, 100, 25))
print(z)
# Plot
bins = bins[hist != 0]
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.xticks(np.round(bins,2), rotation='vertical')
plt.xlim([-0.1, 1.8])
for line in range(1,len(z)):
    plt.axvline(x=z[line], color = 'r', ls = '--')
plt.bar(bins, hist[hist!=0], align='center', width=width, color = '#808080')
plt.xlabel('Score')
plt.ylabel('Distr')
plt.title('ALSFRS')
plt.tick_params(labelsize = 6)
plt.savefig('TimeEvo.png')