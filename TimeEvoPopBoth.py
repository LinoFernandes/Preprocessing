import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats
import os



if sys.platform == "darwin":
    ALSFRS = pd.read_csv('/Users/Lino/PycharmProjects/Preprocessing2/alsfrs.csv')
    ALSFRSR = pd.read_csv('/Users/Lino/PycharmProjects/Preprocessing2/alsfrsr.csv')
    Demographics = pd.read_csv('/Users/Lino/PycharmProjects/Preprocessing2/demo.csv')
elif sys.platform == "win32":
    ALSFRS = pd.read_csv('C:\\Users\\Lino\\PycharmProjects\\Preprocessing2\\alsfrs.csv')
    Demographics = pd.read_csv('C:\\Users\\Lino\\PycharmProjects\\Preprocessing2\\demo.csv')


Demographics = Demographics.values
ALSFRS = ALSFRS.values
ALSFRSR = ALSFRSR.values

strentry = [i for i in range(0, len(ALSFRS)) if ALSFRS[i,1] == 'n' or ALSFRS[i,1] == 'N']
ALSFRS[strentry,1] = np.NaN
strentry = [i for i in range(0, len(ALSFRSR)) if ALSFRSR[i,1] == 'n' or ALSFRSR[i,1] == 'N']
ALSFRSR[strentry,1] = np.NaN

CommonIndex = []
for i in range(0,len(ALSFRS)):
    try:
        int(ALSFRS[i,1])/int(ALSFRSR[i,1])
        CommonIndex.append(i)
    except ValueError:
        continue

ALSFRS = ALSFRS[CommonIndex,:]
ALSFRSR = ALSFRSR[CommonIndex, :]

Score = []
ScoreR = []
for entry in range(0,len(ALSFRS)):
    try:
        ALSFRS[entry, 1] = int(ALSFRS[entry, 1])
        ALSFRSR[entry, 1] = int(ALSFRSR[entry, 1])

        Index = ALSFRS[entry,0]
        Denominator = Demographics[np.where(Demographics[:,0] == Index),1]
        score = (40 - ALSFRS[entry, 1]) / Denominator[0][0]
        Score.append(score)

        score = (48 - ALSFRSR[entry, 1]) / Denominator[0][0]
        ScoreR.append(score)
    except ValueError:
        Score.append(np.NaN)
        ScoreR.append(np.NaN)
Score = np.array(Score)
ScoreR = np.array(ScoreR)


NotNaN = [i for i in range(0, len(Score)) if not np.isnan(Score[i]) or Score[i] > 0]


hist, bins = np.histogram(Score[NotNaN], bins=500)
freq = 5
hist[np.where(hist <= freq)] = 0
z1=Score[NotNaN]
print(len(z1))
z=np.percentile(z1, np.arange(0, 100, 25))
z=np.percentile(Score[NotNaN], np.arange(0, 100, 25))
print(z)
# Plot
# bins = bins[hist != 0]
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(bins, hist[hist!=0], align='center', width=width, color = 'b')
# plt.xticks(np.round(bins,2), rotation='vertical')
# plt.xlim([-0.1, 2.2])
# for line in range(1,len(z)):
#     plt.axvline(x=z[line], color = '#ff0000', mfc = 'r', ls = '--')
# plt.xlabel('Score')
# plt.ylabel('Distr')
# plt.title('ALSFRS')
# plt.tick_params(labelsize=6)
# plt.ylim([0, 43])
# plt.savefig('TimeEvoBoth.png')
# plt.clf()
#
### ALSFRSR
hist, bins = np.histogram(ScoreR[NotNaN], bins=500)
freq = 5
hist[np.where(hist <= freq)] = 0
z1=ScoreR[NotNaN]
z=np.percentile(z1, np.arange(0, 100, 25))
z=np.percentile(ScoreR[NotNaN], np.arange(0, 100, 25))

print(len(z1))
print(z)
# # Plot
# bins = bins[hist != 0]
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(bins, hist[hist!=0], align='center', width=width, color = 'b')
# plt.xticks(np.round(bins,2), rotation='vertical')
# plt.tick_params(labelsize=6)
# plt.xlim([-0.1, 1.7])
# for line in range(1,len(z)):
#     plt.axvline(x=z[line], color='#ff0000', mfc='r', ls='--')
# plt.xlabel('Score')
# plt.ylabel('Distr')
# plt.title('ALSFRS-R')
# plt.savefig('TimeEvoBothR.png')