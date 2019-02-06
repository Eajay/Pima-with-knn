import numpy as np
import csv
from sklearn import preprocessing
from sklearn import decomposition
import matplotlib.pyplot as plt

training_size = 668  # 768
testing_size = 100
k_neighbors = 27

def read_data():
    file = "diabetes.csv"
    c = open(file, "r")
    lines = csv.reader(c)
    header = next(lines)
    data = []
    for line in lines:
        data.append(line)
    data = np.array(data, dtype=float)
    data_labels = list(map(int, data[:, -1].tolist()))
    data = data[:, :-1]
    scaler = preprocessing.StandardScaler()
    # scaler = preprocessing.MinMaxScaler()
    data = scaler.fit_transform(data)

    training_labels = data_labels[:training_size]
    testing_labels = data_labels[training_size:]

    training_data = data[:training_size, :]
    testing_data = data[training_size:, :]

    return training_labels, training_data, testing_labels, testing_data


training_labels, training_data, testing_labels, testing_data = read_data()
pca = decomposition.PCA(n_components=2)
training_data = pca.fit_transform(training_data)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Scatter Plot')
plt.xlabel('value x')
plt.ylabel('value y')
for i in range(training_size):
    if training_labels[i] == 0:
        ax1.scatter(training_data[i][0], training_data[i][1], c='b', marker='o')
    else:
        ax1.scatter(training_data[i][0], training_data[i][1], c='r', marker='o')
label = ["1", "0"]
plt.legend(label)
plt.show()




