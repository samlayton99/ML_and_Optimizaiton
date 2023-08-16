# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
<Sam Layton>
<Vol2 Section 003>
<10/20/22>
"""

import numpy as np
from scipy.spatial import KDTree
import scipy.stats
from matplotlib import pyplot as plt


# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    # Convert X and z to numpy ndarrays
    X = np.array(X)
    z = np.array(z)

    # find the nearest neighbor using broadcasting
    C = X[:, np.newaxis] - z
    dist = np.linalg.norm(C, axis=-1)

    # return the nearest neighbor and its distance
    return X[dist.argmin()], float(min(dist))


# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    def __init__(self, x):
        """Initialize the node with training data x."""\
        # Check if x is already an ndarray, and raise a TypeError if it isn't.
        if type(x) != np.ndarray:
            raise TypeError('Must input an np.ndarray')
    
        # If x is an ndarray, set following attributes equal to None 
        else:
            self.left = None
            self.right = None
            self.pivot = None

            # Store the x given as the node's value
            self.value = x


# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        # make the new node to be inserted
        node = KDTNode(data)

        # If the tree is empty, create a new root node with appropriate k and pivot
        if self.root is None:
            node.pivot = 0
            self.root = node
            self.k = len(data)
        
        # If the data inserted is not in R^k, raise a ValueError
        elif len(node.value) != self.k:
            raise ValueError('Data must be in R^k')
        
        # If the tree is not empty, insert the node in the appropriate place
        else:
            try: 
                if self.find(data) is not None:
                    raise ValueError('Data is already in the tree')
            except:
                pass
            
            def _step(current, parent = None):
                """Recursively step through the tree until the correct node is inserted."""
                # Base case, when the current node is None, insert the new node

                if current is None:
                    # Check if it is a left child at position k
                    if data[parent.pivot] < parent.value[parent.pivot]:
                        parent.left = node
                        node.pivot = (parent.pivot + 1) % self.k
                    # If it is a right child, insert it as a right child
                    else:
                        parent.right = node
                        node.pivot = (parent.pivot + 1) % self.k
                    return node
                
                # Increment k and reset it to 0 if it is greater than k, store as r
               

                # Recursively search left and index k
                if data[current.pivot] < current.value[current.pivot]:
                    return _step(current.left, current)

                # Recursively search right and index k
                else:                                   
                    return _step(current.right, current)
            
            # Start the recursion on the root of the tree.
            _step(self.root)


    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        # Define a step function to recursively search the tree
        def _step(current, nearest, d):

            # Base case, when the current node is None set the distance and value
            if current is None:
                return nearest, d

            # Set x equal to the current node's value
            x = current.value
            i = current.pivot

            # Calculate the distance between z and x and compare it to the distance at this step
            if np.linalg.norm(x - z) < d:
                nearest = current
                d = np.linalg.norm(x - z)

            # search to the left 
            if z[i] < x[i]:
                nearest, d = _step(current.left, nearest, d)

                # search to the right if needed
                if z[i] + d >= x[i]:
                    nearest, d = _step(current.right, nearest, d)

            # search to the right
            else:
                nearest, d = _step(current.right, nearest, d)

                # search to the left if needed
                if z[i] - d <= x[i]:
                    nearest, d = _step(current.left, nearest, d)

            # Return the nearest neighbor and the distance
            return nearest, d

        # run the recursive function on the root and set the return values
        node, d = _step(self.root, self.root, np.linalg.norm(self.root.value - z))
        return node.value, d


    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """
    def __init__(self, n_neighbors):
        """Initialize the classifier.

        Parameters:
            n_neighbors (int): the number of neighbors to use in classification.
        """
        self.k = n_neighbors
        self.kdtree = None

    def fit(self, X, y):
        """Fit the classifier to the training data.

        Parameters:
            X ((n,k) ndarray): the training data.
            y ((n,) ndarray): the training labels.
        """
        # Store the training data and labels.
        self.X = X
        self.y = y

        # Build a KDTree from the training data.
        self.kdtree = KDTree(X)

    def predict(self, z):
        """Predict the label of a new point z.

        Parameters:
            z ((k,) ndarray): a k-dimensional point to predict the label of.

        Returns:
            (int) the predicted label of z.
        """
        # Find the n nearest neighbors of z and take their indices
        distances, indices = self.kdtree.query(z, k = self.k)

        # If there is only 1 neighbor, return the label of that neighbor
        if self.k == 1:
            return self.y[indices]

        # find the labels corresponding to the indices
        else:
            labels = [self.y[i] for i in indices]

        # return the most common label using scipy.stats.mode
        return scipy.stats.mode(labels)[0][0]


# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    # Load the data from the file and store it in appropriate variables
    data = np.load(filename)
    X_train = data["X_train"].astype(float)
    y_train = data["y_train"]
    X_test = data["X_test"].astype(float)
    y_test = data["y_test"]

    # Create a KNeighborsClassifier and fit it to the training data
    decipher = KNeighborsClassifier(n_neighbors)
    decipher.fit(X_train, y_train)

    # Create a list of the predicted labels and initialize a counter
    predictions = [decipher.predict(X_test[i]) for i in range(len(X_test))]
    correct = 0

    # Loop through the predictions and compare them to the test labels
    for i in range(len(predictions)):
        if predictions[i] == y_test[i]:
            correct += 1
    
    # Calculate the accuracy and return it
    accuracy = correct / len(y_test)
    return accuracy


def digitcheck(n):
    #Input an index and pull from the mnist_subset.npz file to display the number
    #then run the KNeighborsClassifier on the number and return the prediction.
    
    # Load the data from the file and store it in appropriate variables
    data = np.load("mnist_subset.npz")
    X_train = data["X_train"].astype(float)
    y_train = data["y_train"]
    X_test = data["X_test"].astype(float)
    y_test = data["y_test"]

    # Create a KNeighborsClassifier and fit it to the training data
    decipher = KNeighborsClassifier(1)
    decipher.fit(X_train, y_train)

    # Print the prediction
    print(decipher.predict(X_test[n]))

    # Display the number
    plt.imshow(X_test[n].reshape((28,28)), cmap="gray")
    plt.show()

digitcheck(60)
