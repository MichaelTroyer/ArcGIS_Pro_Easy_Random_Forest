from random import seed
from random import randrange
from math import sqrt
import pandas as pd


### Random Forest -------------------------------------------------------------


# Create a random subsample from the dataset with replacement
def subsample(dataset, sample_proportion):
    sample = list()
    n_sample = round(len(dataset) * sample_proportion)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Select the best split point for a dataset
def get_best_split(dataset, n_features):
    # Last column is the class ID
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def is_homogeneous(group):
    return True if len({row[-1] for row in group}) == 1 else False


# Create a terminal node value
def to_terminal(group):
    # At a root node - return most frequent class
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Make a prediction with a decision tree
def predict(node, row):
    # less than = left
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            # Terminal nodes are not dicts, but values
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if is_homogeneous(left) or len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_best_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)
    # process right child
    if is_homogeneous(right) or len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_best_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)


# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    root = get_best_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_proportion, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_proportion)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return (trees, predictions)


def predict_dataset(trees, rows):
    return [bagging_predict(trees, row) for row in rows]


### Validation ----------------------------------------------------------------


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def cross_validation_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def cross_validate(dataset, n_folds, max_depth, min_size, sample_proportion, n_trees, n_features):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        # Remove the test fold and flatten the training folds into a single list
        train_set = list(folds)
        train_set.remove(fold)
        train_set = [row for fold in train_set for row in fold]
        test_set = list()
        # Wipe the class ID from the test data copy
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
        trees, predicted = random_forest(
            train_set,
            test_set,
            max_depth,
            min_size,
            sample_proportion,
            n_trees,
            n_features
            )
        actual = [row[-1] for row in fold]
        accuracy = cross_validation_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores


def confusion_matrix(actual, predicted, normalize=False):
    y_actu = pd.Series(actual, name='Actual')
    y_pred = pd.Series(predicted, name='Predicted')
    conf_matx = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    if not normalize:
        return conf_matx
    else:
        return conf_matx / conf_matx.sum(axis=1)