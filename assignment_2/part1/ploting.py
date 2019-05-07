import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

eval_step = 10
crop = 4000/eval_step
max = 10000/eval_step
model = 'RNN'
runs = np.arange(5, 16, 2)
print(runs)
file_dir = r'results/'


MyData = pd.read_csv(file_dir + model + str(runs[0]) + 'results.csv')
MyData = MyData.truncate(before=crop, after=max)
for m in range(1, 3):
    offset  =max * m
    MyTrunks = MyData.truncate(before=crop+offset, after=max+offset)
    MyData = pd.concat([MyData, MyTrunks])

for n in runs[1:]:
    MyData1 = pd.read_csv(file_dir + model + str(n) + 'results.csv')
    MyData1 = MyData1.truncate(before=crop, after=max)
    for m in range(1, 3):
        offset = max * m
        print(offset)
        MyTrunks = MyData1.truncate(before=crop + offset, after=max + offset)
        MyData1 = pd.concat([MyData1, MyTrunks])
    MyData = pd.concat([MyData, MyData1], ignore_index=True)

current_palette = sns.diverging_palette(220, 20, n=len(runs), center='light')
tt = sns.violinplot(x='length', y='accuracy', data=MyData, palette=current_palette)

plt.show()
plt.close()