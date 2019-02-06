import numpy as np
import csv
from pandas import read_csv
import matplotlib.pyplot as plt


data = read_csv("diabetes.csv")
data.hist()
plt.show()
