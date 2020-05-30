import pandas as pd
import matplotlib.pyplot as plt
pnList = pd.read_excel('PreprocessedNewsList.xls')
tmp = {}
newsDict = {}

for date in pnList['Release Date']:
    if date[0:10] in tmp.keys():
        tmp[date[0:10]] += 1
    else:
        tmp[date[0:10]] = 1

tmp_k = sorted(tmp)
for key in tmp_k:
    ww=key[6:7]+key[8:10]
    newsDict[ww] = tmp[key]

print(newsDict)
plt.plot_date(newsDict.keys(), newsDict.values())
plt.show()