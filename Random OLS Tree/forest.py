
from platform import uname
import os
import graphviz
from uuid import uuid4
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time









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
        self.column = column
        self.value = value
        self.features = feature_names[self.column]

    def match(self, sample):
        """Returns T/F depending on how the sample answers the question
        Parameters:
            sample ((n,), ndarray): New sample to classify
        Returns:
            (bool): How the sample compares to the question"""
        # Return True for all samples satisfying the inequality
        return sample[:,self.column] >= self.value
    

def partition(data, question):
    """Splits the data into left (true) and right (false)
    Parameters:
        data ((m,n), ndarray): data to partition
        question (Question): question to split on
    Returns:
        left ((j,n), ndarray): Portion of the data matching the question
        right ((m-j, n), ndarray): Portion of the data NOT matching the question
    """
    # Get the boolean array of which samples match the question and return the split
    left = question.match(data)
    return data[left], data[~left]

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
        return {int(data[-1]) : 1}
    counts = {}
    for label in data[:,-1]:
        if label not in counts:
            counts[int(label)] = 0
        counts[int(label)] += 1
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
    # Define the iterator based on whether or not we're using a random subset
    if random_subset:
        n = int(np.floor(np.sqrt(data.shape[1]-1)))
        iterator = np.random.choice(data.shape[1]-1, n, replace=False)

    # If we're not using a random subset, loop through all columns
    else:
        iterator = range(data.shape[1]-1)

    # Initialize the best gain and question and loop through the first n-1 columns
    best_gain = 0
    best_question = None
    for i in iterator:

        # Loop through each value in the column, making a question for each and partitioning
        for value in data[:,i]:
            question = Question(i, value, feature_names)
            left, right = partition(data, question)

            # If the partition is too small, skip it
            if (num_rows(left) < min_samples_leaf) or (num_rows(right) < min_samples_leaf):
                continue

            # Calculate the info gain and update the best gain and question if necessary
            gain = info_gain(data, left, right)
            if gain > best_gain:
                best_gain, best_question = gain, question

    # Return the best gain and question
    return best_gain, best_question


# Problem 3
class Leaf:
    """Tree leaf node
    Attribute:
        prediction (dict): Dictionary of labels at the leaf"""
    def __init__(self,data):
        # Count up the number of samples of each class label
        self.prediction = class_counts(data)

class Decision_Node:
    """Tree node with a question
    Attributes:
        question (Question): Question associated with node
        left (Decision_Node or Leaf): child branch
        right (Decision_Node or Leaf): child branch"""
    def __init__(self, question, left_branch, right_branch):
        # Set the question and branches
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
    # If the number of rows is less than the minimum samples per leaf, return a leaf
    if data.shape[0] < 2*min_samples_leaf:
        return Leaf(data)
    
    # Find the best question to ask and return a leaf if there is no gain or past max depth
    gain, best_q = find_best_split(data, feature_names, min_samples_leaf, random_subset)
    if gain == 0 or current_depth >= max_depth:
        return Leaf(data)
    
    # Partition the data and build the left and right branches
    left, right = partition(data, best_q)
    left_branch = build_tree(left, feature_names, min_samples_leaf, max_depth, current_depth+1, random_subset)
    right_branch = build_tree(right, feature_names, min_samples_leaf, max_depth, current_depth+1, random_subset)

    # Return a Decision_Node with the best question and branches
    return Decision_Node(best_q, left_branch, right_branch)


# Problem 5
def predict_tree(sample, my_tree):
    """Predict the label for a sample given a pre-made decision tree
    Parameters:
        sample (ndarray): a single sample
        my_tree (Decision_Node or Leaf): a decision tree
    Returns:
        Label to be assigned to new sample"""
    # If the tree is a leaf, return the max prediction
    if isinstance(my_tree, Leaf):
        predictions = my_tree.prediction
        return max(predictions, key=predictions.get)
    
    # If the tree is a node, ask the question and recurse
    if my_tree.question.match(sample):
        return predict_tree(sample, my_tree.left)
    else:
        return predict_tree(sample, my_tree.right)



# Problem 6
def predict_forest(sample, forest):
    """Predict the label for a new sample, given a random forest
    Parameters:
        sample (ndarray): a single sample
        forest (list): a list of decision trees
    Returns:
        Label to be assigned to new sample"""
    # Get the predictions of each tree and return the max
    predictions = [predict_tree(sample, tree) for tree in forest]
    return np.max(predictions)


def generate_forest(data, feature_names, num_trees=10, min_samples_leaf=5, max_depth=4, random_subset=True):
    """Generate a random forest
    Parameters:
        data (ndarray)
        feature_names(list or array)
        num_trees (int): number of trees in the forest
        min_samples_leaf (int): minimum allowed number of samples per leaf
        max_depth (int): maximum allowed depth
        random_subset (bool): whether or not to train on a random subset of features
    Returns:
        (list): list of decision trees"""
    # Initialize the forest
    forest = []

    # Loop through the number of trees and append a tree to the forest
    for i in range(num_trees):
        np.random.shuffle(data)
        tree = build_tree(data, feature_names, min_samples_leaf, max_depth, 0, random_subset)
        forest.append(tree)

    # Return the forest
    return forest

###################################



def analyze_tree(dataset,my_tree):
    """Test how accurately a tree classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        tree (Decision_Node or Leaf): a decision tree
    Returns:
        (float): Proportion of dataset classified correctly"""
    # Initialize number of correct predictions
    correct = 0

    # loop through each sample and get the sample and row
    for row in dataset:
        sample = row[:-1]
        label = row[-1]

        # Get the prediction and update the number of correct predictions
        prediction = predict_tree(sample[np.newaxis,:], my_tree)
        if prediction == label:
            correct += 1

    # Return the proportion of correct predictions
    return correct/dataset.shape[0]



def analyze_forest(dataset,forest):
    """Test how accurately a forest classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        forest (list): list of decision trees
    Returns:
        (float): Proportion of dataset classified correctly"""
    # Initialize number of correct predictions
    correct = 0

    # loop through each sample and get the sample and row
    for row in dataset:
        sample = row[:-1]
        label = row[-1]

        # Get the prediction and update the number of correct predictions
        prediction = predict_forest(sample[np.newaxis,:], forest)
        if prediction == label:
            correct += 1

    # Return the proportion of correct predictions
    return correct/dataset.shape[0]

import pandas as pd
housing = pd.read_csv('housing.csv')

numeric_columns = housing.select_dtypes(include=['number']).columns

# take the remaining columns and convert them to categorical
categorical_columns = housing.columns[~housing.columns.isin(numeric_columns)]
housing[categorical_columns] = housing[categorical_columns].astype('category')

# check if housing is a pandas dataframe or a numpy array


test_df = np.random