
import time
from platform import uname
import os
import graphviz
from uuid import uuid4

import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF


# Problem 1
class Question:
    """Questions to use in construction and display of Decision Trees.
    Attributes:
        column (int): which column of the data this question asks
        value (int/float): value the question asks about
        features (str): name of the feature asked about
    Methods:
        match: returns boolean of if a given sample answered T/F"""

    def __init__(self, column, value, feature_names):
        # Store attributes
        self.column = column
        self.value = value
        self.features = feature_names[self.column]

    def match(self, sample):
        """Returns T/F depending on how the sample answers the question
        Parameters:
            sample ((n,), ndarray): New sample to classify
        Returns:
            (bool): How the sample compares to the question"""
        # Check if it is a match
        return sample[self.column] >= self.value

    def __repr__(self):
        return "Is %s >= %s?" % (self.features, str(float(self.value)))


def partition(data, question):
    """Splits the data into left (true) and right (false)
    Parameters:
        data ((m,n), ndarray): data to partition
        question (Question): question to split on
    Returns:
        left ((j,n), ndarray): Portion of the data matching the question
        right ((m-j, n), ndarray): Portion of the data NOT matching the question
    """
    n = data[0].shape[0]

    # Get the feature
    feature = data[:, question.column]

    mask = feature >= question.value

    # Mask it to get the left and right partitions
    left = data[mask]
    right = data[~mask]

    return left.reshape(-1, n), right.reshape(-1, n)


# Helper function
def num_rows(array):
    """ Returns the number of rows in a given array """
    if array is None:
        return 0
    elif len(array.shape) == 1:
        return 1
    else:
        return array.shape[0]


# Helper function
def class_counts(data):
    """ Returns a dictionary with the number of samples under each class label
        formatted {label : number_of_samples} """
    if len(data.shape) == 1: # If there's only one row
        return {data[-1] : 1}
    counts = {}
    for label in data[:,-1]:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

# Helper function
def info_gain(data, left, right):
    """Return the info gain of a partition of data.
    Parameters:
        data (ndarray): the unsplit data
        left (ndarray): left split of data
        right (ndarray): right split of data
    Returns:
        (float): info gain of the data"""
        
    def gini(data):
        """Return the Gini impurity of given array of data.
        Parameters:
            data (ndarray): data to examine
        Returns:
            (float): Gini impurity of the data"""
        counts = class_counts(data)
        N = num_rows(data)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / N
            impurity -= prob_of_lbl**2
        return impurity
        
    p = num_rows(right)/(num_rows(left)+num_rows(right))
    return gini(data) - p*gini(right)-(1-p)*gini(left)

# Problem 2, Problem 6
def find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False):
    """Find the optimal split
    Parameters:
        data (ndarray): Data in question
        feature_names (list of strings): Labels for each column of data
        min_samples_leaf (int): minimum number of samples per leaf
        random_subset (bool): for Problem 6
    Returns:
        (float): Best info gain
        (Question): Best question"""
    # Initialize variables
    best_gain = 0
    best_question = None

    m, n = data.shape
    n -= 1
    feature_indices = np.arange(n)

    if random_subset:  # If it is random
        num_feat = int(np.sqrt(n))
        feature_indices = np.random.choice(feature_indices, num_feat, replace=False)

    # Iterate through each feature (column) (Do not iterate through final column
    for j in feature_indices:
        # Iterate through each unique value (row)
        for i in range(m):
            # Create Question obj with column and value
            question = Question(j, data[i, j], feature_names=feature_names)

            # Use partition to split the dataset into left and right partitions
            left, right = partition(data, question)

            # If either left or right partitions are smaller than allowable leaf size reject
            if num_rows(left) < min_samples_leaf:
                continue
            elif num_rows(right) < min_samples_leaf:
                continue

            # Calculate Info Gain
            gain = info_gain(data, left, right)

            # Update best_gain and best_question
            if gain > best_gain:
                best_gain = gain
                best_question = question

    return best_gain, best_question


# Problem 3
class Leaf:
    """Tree leaf node
    Attribute:
        prediction (dict): Dictionary of labels at the leaf"""
    def __init__(self, data):
        # Store attributes
        self.prediction = class_counts(data)


class Decision_Node:
    """Tree node with a question
    Attributes:
        question (Question): Question associated with node
        left (Decision_Node or Leaf): child branch
        right (Decision_Node or Leaf): child branch"""
    def __init__(self, question, left_branch, right_branch):
        # Store attributes
        self.question = question
        self.left = left_branch
        self.right = right_branch


# Prolem 4
def build_tree(data, feature_names, min_samples_leaf=5, max_depth=4, current_depth=0, random_subset=False):
    """Build a classification tree using the classes Decision_Node and Leaf
    Parameters:
        data (ndarray)
        feature_names(list or array)
        min_samples_leaf (int): minimum allowed number of samples per leaf
        max_depth (int): maximum allowed depth
        current_depth (int): depth counter
        random_subset (bool): whether or not to train on a random subset of features
    Returns:
        Decision_Node (or Leaf)"""

    # Check if there are enough rows left
    if num_rows(data) < 2 * min_samples_leaf:
        return Leaf(data)

    # Find optimal info gain and question
    gain, question = find_best_split(data,
        feature_names=feature_names,
        min_samples_leaf=min_samples_leaf,
        random_subset=random_subset)

    # Check base case
    if gain == 0 or current_depth >= max_depth:
        return Leaf(data)

    # Call recursively
    left, right = partition(data, question)
    left_tree = build_tree(left, feature_names, min_samples_leaf, max_depth, current_depth+1, random_subset)
    right_tree = build_tree(right, feature_names, min_samples_leaf, max_depth, current_depth+1, random_subset)

    return Decision_Node(question, left_tree, right_tree)


# Problem 5
def predict_tree(sample, my_tree):
    """Predict the label for a sample given a pre-made decision tree
    Parameters:
        sample (ndarray): a single sample
        my_tree (Decision_Node or Leaf): a decision tree
    Returns:
        Label to be assigned to new sample"""

    # Base case if my_tree is a leaf
    if isinstance(my_tree, Leaf):
        return max(my_tree.prediction, key=my_tree.prediction.get)

    # Otherwise break it down into left and right branches
    if my_tree.question.match(sample):
        return predict_tree(sample, my_tree.left)
    else:
        return predict_tree(sample, my_tree.right)


def analyze_tree(dataset, my_tree):
    """Test how accurately a tree classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        tree (Decision_Node or Leaf): a decision tree
    Returns:
        (float): Proportion of dataset classified correctly"""

    # Get labels
    actual_labels = dataset[:, -1]

    n = dataset.shape[0]

    # Predict each sample in dataset
    predicted_labels = np.array(
        [predict_tree(dataset[i], my_tree) for i in range(n)])

    return (actual_labels == predicted_labels).sum() / n


# Problem 6
def predict_forest(sample, forest):
    """Predict the label for a new sample, given a random forest
    Parameters:
        sample (ndarray): a single sample
        forest (list): a list of decision trees
    Returns:
        Label to be assigned to new sample"""
    n = len(forest)

    # Predict the sample label using each tree
    predicted_labels = np.array(
        [predict_tree(sample, forest[i]) for i in range(n)]).astype(int)

    return np.argmax(np.bincount(predicted_labels))


def analyze_forest(dataset, forest):
    """Test how accurately a forest classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        forest (list): list of decision trees
    Returns:
        (float): Proportion of dataset classified correctly"""

    # Get labels
    actual_labels = dataset[:, -1]

    n = dataset.shape[0]

    # Predict each sample in the dataset
    predicted_labels = np.array(
        [predict_forest(dataset[i], forest) for i in range(n)])

    return (actual_labels == predicted_labels).sum() / n


# Problem 7
def prob7():
    """ Using the file parkinsons.csv, return three tuples. For tuples 1 and 2,
        randomly select 130 samples; use 100 for training and 30 for testing.
        For tuple 3, use the entire dataset with an 80-20 train-test split.
        Tuple 1:
            a) Your accuracy in a 5-tree forest with min_samples_leaf=15
                and max_depth=4
            b) The time it took to run your 5-tree forest
        Tuple 2:
            a) Scikit-Learn's accuracy in a 5-tree forest with
                min_samples_leaf=15 and max_depth=4
            b) The time it took to run that 5-tree forest
        Tuple 3:
            a) Scikit-Learn's accuracy in a forest with default parameters
            b) The time it took to run that forest with default parameters
    """
    # Get data and features
    dataset = np.loadtxt('parkinsons.csv', delimiter=',')[:, 1:]
    features = np.loadtxt('parkinsons_features.csv', delimiter=',', dtype=str)
    total = dataset.shape[0]

    # Get random sample of the data
    random_indices = np.random.choice(np.arange(dataset.shape[0]), 130, replace=False)
    random_data = dataset[random_indices]

    # Split the sample into a training and testing set
    random_indices = np.random.choice(np.arange(130), 100, replace=False)
    X_train = random_data[random_indices]
    X_test = random_data[np.setdiff1d(np.arange(130), random_indices)]

    # Split the entire dataset into training and testing set
    per80 = int(total * 0.8)
    random_indices = np.random.choice(np.arange(total), per80, replace=False)
    X_train_full = dataset[random_indices]
    X_test_full = dataset[np.setdiff1d(np.arange(total), random_indices)]

    # Set parameters as mentioned
    min_samples_leaf = 15
    max_depth = 4

    # Time my implementation
    start = time.perf_counter()

    my_forest = [build_tree(
        X_train,
        features,
        min_samples_leaf,
        max_depth) for _ in range(5)]

    my_acc = analyze_forest(X_test, my_forest)

    end = time.perf_counter()

    my_time = end - start

    # Time scikit-learn implementation
    start = time.perf_counter()

    rf_classifier = RF(n_estimators=5,
                       max_depth=max_depth,
                       min_samples_leaf=min_samples_leaf)
    rf_classifier.fit(X_train[:, :-1], X_train[:, -1])
    sk_acc = rf_classifier.score(X_test[:, :-1], X_test[:, -1])

    end = time.perf_counter()

    sk_time = end - start

    # Time scikit-learn implementation on the full dataset
    start = time.perf_counter()

    rf_classifier = RF(n_estimators=5,
                       max_depth=max_depth,
                       min_samples_leaf=min_samples_leaf)
    rf_classifier.fit(X_train_full[:, :-1], X_train_full[:, -1])
    sk_acc_full = rf_classifier.score(X_test_full[:, :-1], X_test_full[:, -1])

    end = time.perf_counter()

    sk_time_full = end - start

    return (my_acc, my_time), (sk_acc, sk_time), (sk_acc_full, sk_time_full)


# Code to draw a tree
def draw_node(graph, my_tree):
    """Helper function for drawTree"""
    node_id = uuid4().hex
    # If it's a leaf, draw an oval and label with the prediction
    if not hasattr(my_tree, "question"):  # isinstance(my_tree, leaf_class):
        graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
        return node_id
    else:  # If it's not a leaf, make a question box
        graph.node(node_id, shape="box", label="%s" % my_tree.question)
        left_id = draw_node(graph, my_tree.left)
        graph.edge(node_id, left_id, label="T")
        right_id = draw_node(graph, my_tree.right)
        graph.edge(node_id, right_id, label="F")
        return node_id


def draw_tree(my_tree, filename='Digraph', leaf_class=Leaf):
    """Draws a tree"""
    # Remove the files if they already exist
    for file in [f'{filename}.gv',f'{filename}.gv.pdf']:
        if os.path.exists(file):
            os.remove(file)
    graph = graphviz.Digraph(comment="Decision Tree")
    draw_node(graph, my_tree)
    # graph.render(view=True) #This saves Digraph.gv and Digraph.gv.pdf
    in_wsl = False
    in_wsl = 'microsoft-standard' in uname().release
    if in_wsl:
        graph.render(f'{filename}.gv', view=False)
        os.system(f'cmd.exe /C start {filename}.gv.pdf')
    else:
        graph.render(view=True)

