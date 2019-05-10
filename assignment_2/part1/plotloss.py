import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

eval_step = 10
model = 'RNN'
runs = np.arange(7, 10)
print(runs)
file_dir = r'results/'

MyData = pd.read_csv(file_dir + model + str(runs[0]) + 'results.csv')
for n in runs[1:]:
    MyData1 = pd.read_csv(file_dir + model + str(n) + 'results.csv')
    MyData = pd.concat([MyData, MyData1], ignore_index=True)

# smoothing
MyData['epoch'] = MyData['epoch'].apply(lambda step : step - step%100)

# set plot colors
current_palette = sns.diverging_palette(220, 20, n=len(runs), center='dark')

# plot
tt = sns.lineplot(x='epoch', y='loss', hue='length', ci="sd", data=MyData, palette=current_palette)

plt.show()
plt.close()