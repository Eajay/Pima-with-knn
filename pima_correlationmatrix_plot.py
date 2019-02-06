
names = ['pregnancies', 'glucose', 'blood pressure', 'skin thickness', 'insulin', 'BMI', 'diabetes pedigree function', 'age', 'outcome']
names = ['preg', 'glu', 'bl_pr', 'skin', 'insu', 'BMI', 'dpf', 'age', 'out']
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

filename = 'diabetes.csv'
data = read_csv(filename, names=names, header=1)
correlations = data.corr()
plt.subplots(figsize=(9, 9))
sns.heatmap(correlations, annot=True, vmax=1, square=True, cmap="Blues")



# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(correlations, vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0,9,1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(names)
# ax.set_yticklabels(names)
plt.savefig("C:\\Users\\WIN7\\Desktop\\ml\\project1\\correlation_matrix.png")
plt.show()

