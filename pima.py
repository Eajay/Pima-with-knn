import numpy as np
import csv
from sklearn import preprocessing
import time

training_size = 615  # 768
testing_size = 153
k_neighbors = 17

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
    #scaler = preprocessing.MinMaxScaler()
    data = scaler.fit_transform(data)

    training_labels = data_labels[:training_size]
    testing_labels = data_labels[training_size:]

    training_data = data[:training_size, :]
    testing_data = data[training_size:, :]


    return training_labels, training_data, testing_labels, testing_data


training_labels, training_data, testing_labels, testing_data = read_data()


def calculate_k_nearest_neighbors(training_labels, training_data, testing_labels, testing_data):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(testing_size):
        print("test ", i, ":")
        total_start = time.time()
        start = time.time()
        dist = np.sqrt(np.sum(np.square(testing_data[i] - training_data), axis=1))
        end = time.time()
        print("calculate distance costs: ", end - start, "s")
        start = time.time()
        nearest_nums = np.argsort(dist)
        count = [0] * 10
        for j in range(k_neighbors):
            count[training_labels[nearest_nums[j]]] += 1
        guess_result = count.index(max(count))
        end = time.time()
        total_end = time.time()
        real_result = testing_labels[i]
        print("calculate k nearest neighbors costs: ", end - start, "s")
        print("the total cost in this test: ", total_end - total_start, "s")
        print("The real result is {},  the guess result is {}".format(real_result, guess_result))
        print("-----------------------------------")
        if real_result == 1 and guess_result == 1:
            true_positive += 1
        elif real_result == 0 and guess_result == 0:
            true_negative += 1
        elif real_result == 1 and guess_result == 0:
            false_negative += 1
        else:
            false_positive += 1
    print("true positive: ", true_positive)
    print("true negative: ", true_negative)
    print("false positive: ", false_positive)
    print("false negative: ", false_negative)
    print("True Positive Rate: ", true_positive/(true_positive+false_negative))
    print("True Negative Rate: ", true_negative/(true_negative+false_positive))

calculate_k_nearest_neighbors(training_labels, training_data, testing_labels, testing_data)

