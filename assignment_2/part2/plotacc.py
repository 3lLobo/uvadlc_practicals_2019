import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

eval_step = 10
model = 'LSTM'

MyData = pd.read_csv('results02/GEN90tunedlstm.csv')

# smoothing
MyData['Step'] = MyData['step'].apply(lambda step : step - step%1000)
MyData['Accuracy'] = MyData['accuracy']
# set plot colors
current_palette = sns.diverging_palette(220, 20, n=1, center='dark')

# plot
tt = sns.lineplot(x='Step', y='Accuracy', hue='length', data=MyData, ci="sd", palette=current_palette)

plt.show()
plt.close()