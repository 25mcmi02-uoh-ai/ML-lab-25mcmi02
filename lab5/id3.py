# implementation of ID3 algorithm for decision tree learning
import numpy as np
import pandas as pd

def build_tree(X, y):
    # check if all labels are the same
    if len(set(y)) == 1:
        return y[0]

    # find the best split
    best_feature = _best_feature(X, y)

    # Create a new tree node
    tree = {best_feature: {}}

    # Split the dataset by the best feature
    for value in set(X[best_feature]):
        subset_X = X[X[best_feature] == value].drop(columns=[best_feature])
        subset_y = y[X[best_feature] == value]
        tree[best_feature][value] = build_tree(subset_X, subset_y)

    return tree

def _best_feature(X, y):
    # get information gain for each feature then choose max
    best_gain = -1
    best_feature = None

    for feature in X.columns:
        new_gain = _information_gain(X[feature], y)
        if new_gain > best_gain:
            best_gain = new_gain
            best_feature = feature

    return best_feature


def _information_gain(feature, y):
    total_entropy = _entropy(y)
    weighted_entropy = 0
    for v in set(feature):
        weighted_entropy = weighted_entropy + (len(y[feature == v]) / len(y)) * _entropy(y[feature == v])

    return total_entropy - weighted_entropy


# def _entropy(y):
#     probabilities = np.bincount(y) / len(y)
#     return -np.sum(probabilities * np.log2(probabilities + 1e-10))

# implementation of entropy that works with non-numeric labels
def _entropy(y):
    probabilities = y.value_counts(normalize=True)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def gini_impurity(labels):
    n = len(labels)
    classes = np.unique(labels)
    gini = 0.0
    for c in classes:
        p = np.sum(labels == c)

if __name__ == "__main__":
    df = pd.read_csv("data.csv", sep=" ")
    X = df[["Weather", "Parents", "Cash", "Exam"]]
    y = df["Decision"]
    tree = build_tree(X, y)
    print(tree)