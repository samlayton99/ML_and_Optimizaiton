import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


class model3():
    def __init__(self, max_iter=1000, tol=1e-8, learning_rate=0.01, reg = 10, dim_reg = 0, stochastic=False, batch_size=32, epochs = 100):
        # Hyperparameters
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.stochastic = stochastic
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg = reg
        self.dim_reg = dim_reg

        # Variables to store
        self.X = None
        self.y = None
        self.n = None
        self.d = None
        self.weights = None
        self.classes = None
        self.num_classes = None
        self.y_dict = None
        self.one_hot = None
        self.differences = None
        self.X_val_set = None
        self.y_val_set = None
        self.class_index = []
        self.val_history = []
        self.train_history = []
        self.weights_history = []
        

    ############################## Helper Functions ###############################
    def label_from_stationary(self, stationary, show_probabilities = False):
        """Get the label from the stationary distribution
        Input:
            stationary (n,) ndarray - The stationary distribution of the data
        Output:
            labels (n,) ndarray - The predicted labels of the data"""
        
        # Check if the sum of the stationary distribution is approximately 1
        if not np.isclose(np.sum(stationary), 1):
            raise ValueError("The weights do not sum to 1.")
        
        # Sum the weights for each class
        class_probabilities = np.zeros(self.num_classes)
        for weight, label in zip(stationary, self.y):
            class_probabilities[self.y_dict[label]] += weight

        # Return the probabilities if requested
        if show_probabilities:
            return class_probabilities
        
        # Otherwise, return the class with the highest weight
        else:
            indices = np.argmax(class_probabilities)
            return self.classes[indices]
        
    def get_gaussian(self, weights):
        """ Get the gaussian kernel for the informative points
        Input:
            informative_points (n,d) ndarray - The informative points
            target (n,) ndarray - The target values
            weights (d,) ndarray - The weights for the informative points
        Output:
            gaussian (n,n) ndarray - The gaussian kernel for the informative points"""
        tensor_prod = np.einsum('ijk,lk->ijl', self.differences, weights)
        return np.exp(-np.linalg.norm(tensor_prod, axis=2)).T, tensor_prod
    

    def encode_y(self,y):
        # Check if the input is a list
        if isinstance(y, list):
            y = np.array(y)

        # Make sure it is a numpy array
        elif not isinstance(y, np.ndarray):
            raise ValueError("y must be a list or a numpy array")
        
        # If it is not integers, give it a dictionary
        if y.dtype != int:
            self.classes = np.unique(y)
            self.num_classes = len(self.classes)
            self.y_dict = {label: i for i, label in enumerate(np.unique(y))}

        # If it is, still make it a dictionary
        else:
            self.classes = np.arange(np.max(y)+1)
            self.num_classes = len(self.classes)
            self.y_dict = {i: i for i in self.classes}

        # Create an index array
        for i in range(self.num_classes):
            self.class_index.append(np.where(y == self.classes[i])[0])

        # Make a one hot encoding
        self.one_hot = np.zeros((self.n, self.num_classes))
        for i in range(self.n):
            self.one_hot[i, self.y_dict[y[i]]] = 1


    ############################## Training Functions ##############################
    def gradient(self, W):
        # Get the gaussian kernel, and tensor product
        gaussian, tensor_prod = self.get_gaussian(W) 
        dW = np.zeros((self.d,self.d))

        # Loop through the different classes and select the right subsets
        for i in range(len(self.classes)):
            g_c = gaussian[self.class_index[i]]
            g_c_totals = np.sum(g_c, axis=0)[:, np.newaxis, np.newaxis] + 1e-20
            product_c = tensor_prod[self.class_index[i]]
            differences_c = self.differences[self.class_index[i]]

            # Calculate the weighted products
            weighted_product_c = g_c[:,:,np.newaxis] * product_c
            weighted_sum_c = np.sum(np.einsum('ijk,ijl->ijkl', weighted_product_c, differences_c), axis = 0)
            weighted_sum_c /= g_c_totals

            # Calculate the gradient first term
            dW += np.sum(self.one_hot[:,i][:,np.newaxis, np.newaxis] * weighted_sum_c, axis=0)

        # Calculate the gradient first term
        g_all_totals = np.sum(gaussian, axis=0)[:, np.newaxis, np.newaxis] + 1e-20
        weighted_product_all = gaussian[:,:,np.newaxis] * tensor_prod
        weighted_sum_all = np.sum(np.einsum('ijk,ijl->ijkl', weighted_product_all, self.differences), axis = 0)
        weighted_sum_all /= g_all_totals

        # Calculate the gradient second term
        dW -= np.sum(weighted_sum_all, axis=0)

        # Return the regularized gradient
        return 2*dW + self.reg*(W / np.linalg.norm(W, 'fro') - self.dim_reg*np.eye(self.d) /self.d)

    def gradient_descent(self):
        show_iter = max(self.max_iter,100) // 100
        for i in range(self.max_iter):
            # Get the gradient
            gradient = self.gradient(self.weights)
            self.weights -= self.learning_rate * gradient
            self.weights_history.append(self.weights.copy())

            # If there is a validation set, check the validation error
            if self.X_val_set is not None and self.y_val_set is not None:
                
                # Predict on the validation set and append the history
                val_predictions = self.predict(self.X_val_set)
                val_accuracy = accuracy_score(self.y_val_set, val_predictions)
                self.val_history.append(val_accuracy)

                # Predict on the training set and append the history
                train_predictions = self.predict(self.X)
                train_accuracy = accuracy_score(self.y, train_predictions)
                self.train_history.append(train_accuracy)
                
                # Show the progress
                if i % show_iter == 0:
                    print(f"({i}) Val Accuracy: {np.round(val_accuracy,5)}.   Train Accuracy: {train_accuracy}")
                    # plt.imshow(self.weights)
                    # plt.show()

            # Check for convergence after a certain number of iterations
            break_value = np.linalg.norm(gradient)
            if break_value < self.tol*self.n and i > self.max_iter//50:
                break

    def stochastic_gradient_descent(self):
        pass

    def fit(self, X, y, X_val_set = None, y_val_set = None):
        # Save the data as variables
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.weights = .125*((np.random.random((self.d,self.d))*2 - 1) + np.eye(self.d))
        self.encode_y(y)

        # If there is a validation set, save it
        if X_val_set is not None and y_val_set is not None:
            self.X_val_set = X_val_set
            self.y_val_set = y_val_set

        # Perform necessary calculations
        self.differences = self.X[:,np.newaxis,:] - self.X[np.newaxis,:,:]

        # If it is not stochastic, run the gradient descent
        if not self.stochastic:
            self.gradient_descent()

        # Otherwise, run the stochastic gradient descent
        else:
            self.stochastic_gradient_descent()

    

    ############################## Prediction Functions #############################
    def predict(self, points, show_probabilities=False):
        """Predict the labels of the data
        Input:
            points (n,d) ndarray - The data to predict
        Output:
            predictions (n,) ndarray - The predicted labels of the data"""
        # Check the shape of the data and the point and initialize the predictions
        if len(points.shape) == 1:
            points = points[np.newaxis,:]
        predictions = []

        # Get the differences array
        differences = self.X[:,np.newaxis,:] - points[np.newaxis,:,:]
        probs = np.exp(-np.linalg.norm(np.einsum('ijk,lk->ijl', differences, self.weights), axis=2)).T + 1e-75
        probs /= np.sum(probs, axis=1, keepdims=True)
        
        # Loop through the different points and get the predictions
        for i in range(points.shape[0]):
            predictions.append(self.label_from_stationary(probs[i], show_probabilities=show_probabilities))

        # Return the predictions
        return np.array(predictions)


    ############################## Analysis Functions ###############################
    def copy(self):
        return self
        