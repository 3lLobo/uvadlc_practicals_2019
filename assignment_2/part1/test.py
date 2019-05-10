import pandas as pd

# save with pandas too

acc_max = [1,3,4,5]
header = ['accuracy', 'lenght']
savefiles = zip(acc_max, [1]*len(acc_max))
df = pd.DataFrame(savefiles, columns=header)
filedir = r'./results/'
df.to_csv(filedir + 'test.csv')
print('Done training.')

