import csv
from collections import Counter, OrderedDict
from typing import List


class Node:
    def __init__(self, value, left, right):
        self.value = value
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None


# Implementation of Algorithm 1 - DecisionTreeTrain
# example data:
# [OrderedDict([('target', 'LIKED'), ('easy', 'y'), ('AI', 'y'), ('sys', 'n'), ('thy', 'y'), ('morning', 'n')])]
def decision_tree_train(data: List[OrderedDict], remaining_features: set):
    targets = [example['target'] for example in data]

    if not targets:
        return None

    # guess ← most frequent answer in data
    counter = Counter(targets)
    guess = counter.most_common(1)[0][0]

    # unambiguous or no remaining features
    if len(counter) == 1 or not remaining_features:
        return Node(guess, None, None)

    score = dict()

    for feature in remaining_features:
        no = [example for example in data if example[feature] == 'n']
        yes = [example for example in data if example[feature] == 'y']

        majority_vote_answers_no = Counter([example['target'] for example in no]).most_common(1)[0][1] if no else 0
        majority_vote_answers_yes = Counter([example['target'] for example in yes]).most_common(1)[0][1] if yes else 0

        score[feature] = majority_vote_answers_no + majority_vote_answers_yes

    max_scoring_feature = max(score, key=score.get)

    no = [example for example in data if example[max_scoring_feature] == 'n']
    yes = [example for example in data if example[max_scoring_feature] == 'y']

    left = decision_tree_train(no, remaining_features - {max_scoring_feature})
    right = decision_tree_train(yes, remaining_features - {max_scoring_feature})
    return Node(max_scoring_feature, left, right)


# Implementation of Algorithm 2 - DecisionTreeTest
def decision_tree_test(node: Node, features: set, example):
    if node.is_leaf():
        return node.value

    if node.value in features:
        if example[node.value] == 'n':
            return decision_tree_test(node.left, features, example)
        else:
            return decision_tree_test(node.right, features, example)


def get_data(path_to_data='/home/arvind/ciml/data/courses_data_train.csv'):
    with open(path_to_data) as f:
        reader = csv.DictReader(f)
        features = set(reader.fieldnames)

        if 'target' in features:
            features.remove('target')
        if 'rating' in features:
            features.remove('rating')

        data = [row for row in reader]
        return data, features


def get_accuracy(train_data, test_data, features):
    decision_tree = decision_tree_train(train_data, features)

    correct_predictions = 0
    for data_point in test_data:
        prediction = decision_tree_test(decision_tree, features, data_point)
        if data_point['target'] == prediction:
            correct_predictions = correct_predictions + 1

    return (float(correct_predictions) / len(train_data))


data, features = get_data()
training_accuracy = get_accuracy(data, data, features)
print('Training accuracy: {}'.format(training_accuracy))
