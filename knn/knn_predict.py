import heapq
from math import sqrt
from collections import Counter

from sklearn import datasets


def get_euclidean_distance(example_1, example_2):
    return sqrt(sum((a - b)**2 for a, b in zip(example_1, example_2)))


def _distance_key(example):
    return example[0]


def knn_predict(train_data, train_targets, test_example, k):
    distances = [(get_euclidean_distance(train_example, test_example), target) for train_example, target in zip(train_data, train_targets)]
    k_nearest = heapq.nsmallest(k, distances, key=_distance_key)
    return Counter(example[1] for example in k_nearest).most_common(1)[0][0]


def get_accuracy(train_data, train_targets, test_data, test_targets, k):
    correct_predictions = 0
    for data_point, target in zip(test_data, test_targets):
        prediction = knn_predict(train_data, train_targets, data_point, k)
        if target == prediction:
            correct_predictions = correct_predictions + 1

    return float(correct_predictions) / len(train_data)


x, y = datasets.load_iris(return_X_y=True)

# trying for different odd values of k, starting from 3
for k in range(3, 10, 2):
    training_accuracy = get_accuracy(x, y, x, y, k)
    print('Training accuracy with {} nearest neighbours: {}'.format(k, training_accuracy))
