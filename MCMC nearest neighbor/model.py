# Class dependencies
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# Other analysis libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


"""
To import this class into an ipynb file in the same folder:

from model import model3    
"""



class model3():
    def __init__(self, max_iter=1000, tol=1e-8, learning_rate=0.01, reg=10, dim_reg=0, optimizer="grad", batch_size=32, epochs=100):
        """Initialize the model
        
        Parameters:
            max_iter (int) - The maximum number of iterations to perform in the gradient descent optimization
            tol (float) - The tolerance for the amount of improvement between iterations during gradient descent optimization
            learning_rate (float) - The learning rate for the gradient descent optimization
            reg (float) - The regularization parameter for learning the covariance matrix
            dim_reg (float) - The regularization parameter for the dimension of the covariance matrix
                            This regularizer adds the identity matrix scaled by dim_reg to the covariance matrix to penalize
                            the model for dropping dimensions
            optimizer (str) - The optimizer to use. 
                            Options are 'grad' for gradient descent, 'sgd' for stochastic gradient descent, and 'bfgs' for BFGS
            batch_size (int) - The batch size for stochastic gradient descent
            epochs (int) - The number of epochs for stochastic gradient descent

        Attributes:
            X (n,d) ndarray - The data to fit the model on
            y (n,) ndarray - The labels of the data
            n (int) - The number of data points
            d (int) - The dimension of the data
            weights (d,d) ndarray - The weights to learn
            classes (num_classes,) ndarray - The classes of the data
            num_classes (int) - The number of classes
            y_dict (dict) - A dictionary mapping the labels to integers
            one_hot (n,num_classes) ndarray - The one hot encoding of the labels
            differences (n,n,d) ndarray - The differences array for the informative points
            cur_gaussian (n,n) ndarray - The current gaussian kernel for the informative points
            cur_tensor_prod (n,n,d) ndarray - The current tensor product for the informative points
            subset_differences (list) - A list of differences arrays for each subset
            X_val_set (n_val,d) ndarray - The validation set for the data
            y_val_set (n_val,) ndarray - The validation labels for the data
            class_index (list) - A list of indices for each class
            val_history (list) - A list of validation accuracies
            train_history (list) - A list of training accuracies
            weights_history (list) - A list of the weights at each iteration
        Returns:
            None
        """
        # Hyperparameters
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.optimizer = optimizer
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
        self.cur_gaussian = None
        self.cur_tensor_prod = None
        self.subset_differences = []

        # Validation variables
        self.X_val_set = None
        self.y_val_set = None
        self.class_index = []
        self.val_history = []
        self.train_history = []
        self.weights_history = []
        

    ############################## Helper Functions ###############################
    def label_from_stationary(self, stationary:np.ndarray, show_probabilities=False):
        """
        Get the label for each class from the stationary distribution.

        Parameters:
            stationary (n,) ndarray - The stationary distribution of the data
            show_probabilities (bool) - Whether to return the probabilities of the classes
        Returns:
            labels (n,) ndarray - The predicted labels of the data 
            OR
            class_probabilities (num_classes,) ndarray - The probabilities of each class based on the stationary distribution
        """
        # Runtime check to ensure `stationary` is a 1D ndarray
        if not isinstance(stationary, np.ndarray):
            raise TypeError("stationary must be a numpy ndarray.")
        if stationary.ndim != 1:
            raise ValueError("stationary must be a 1D ndarray.")
        # Check if the sum of the stationary distribution is approximately 1
        if not np.isclose(np.sum(stationary), 1):
            raise ValueError("The weights do not sum to 1.")
        if not type(show_probabilities) == bool:
            raise TypeError("show_probabilities must be a boolean.")
        
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


    def encode_y(self, y:np.ndarray):
        """
        Encode the labels of the data.

        Parameters:
            y (n,) ndarray - The labels of the data
        Returns:
            None
        """
        # Check if the input is a list
        if isinstance(y, list):
            y = np.array(y)

        # Make sure it is a numpy array
        elif not isinstance(y, np.ndarray):
            raise ValueError("y must be a list or a numpy array")
        
        # If it is not integers, give it a dictionary
        if y.dtype != int:
            self.classes = np.unique(y)
            self.y_dict = {label: i for i, label in enumerate(np.unique(y))}

        # If it is, still make it a dictionary
        else:
            self.classes = np.arange(np.max(y)+1)
            self.y_dict = {i: i for i in self.classes}
        self.num_classes = len(self.classes)

        # Create an index array
        for i in range(self.num_classes):
            self.class_index.append(np.where(y == self.classes[i])[0])

        # Make a one hot encoding
        self.one_hot = np.zeros((self.n, self.num_classes))
        for i in range(self.n):
            self.one_hot[i, self.y_dict[y[i]]] = 1

        
    def randomize_batches(self):
        """
        Randomize the batches for stochastic gradient descent
        Parameters:
            None
        Returns:
            batches (list) - A list of batches of indices for training
        """
        # Get randomized indices and calculate the number of batches
        indices = np.arange(self.n)
        np.random.shuffle(indices)
        num_batches = self.n // self.batch_size

        # Loop through the different batches and get the batches
        batches = [indices[i*self.batch_size:(i+1)*self.batch_size].tolist() for i in range(num_batches)]

        # Handle the remaining points
        remaining_points = indices[num_batches*self.batch_size:]
        counter = len(remaining_points)
        i = 0

        # Fill the remaining points into the batches
        while counter > 0:
            batches[i % len(batches)].append(remaining_points[i])
            i += 1
            counter -= 1

        # Return the batches
        return batches


    ############################## Training Calculations ##############################
    def update_differences(self, X:np.ndarray, batches=None):
        """
        Get the array of difference between the informative points
        
        Parameters:
            X (n,d) ndarray - The data to calculate the differences for
            batches (list) - A list of batches of indices for training
            NOT?
            informative_points (n,d) ndarray - The informative points
        Returns:
            None
            NOT?
            differences (n,n,d) ndarray - The differences array for the informative points
        """
        # If it is not a batch, calculate the differences
        if batches is None:
            self.differences = X[:,np.newaxis,:] - X[np.newaxis,:,:]
        
        # Otherwise, calculate the differences for each batch
        else:
            self.subset_differences = [X[batch][:,np.newaxis,:] - X[batch][np.newaxis,:,:] for batch in batches]

    
    def update_gaussian(self, weights:np.ndarray, subset_num=None):
        """
        Get the gaussian kernel for the informative points
        
        Parameters:
            weights (d,) ndarray - The weights for the informative points
            NOT?
            informative_points (n,d) ndarray - The informative points
            target (n,) ndarray - The target values
        Returns:
            gaussian (n,n) ndarray - The gaussian kernel for the informative points"""
        # If there is no subset, let the differences be the self.differences
        if subset_num is None:
            differences = self.differences
        else:
            differences = self.subset_differences[subset_num]

        # Calculate the gaussian kernel and the tensor product, and save them
        tensor_prod = np.einsum('ijk,lk->ijl', differences, weights)
        self.cur_gaussian = np.exp(-np.linalg.norm(tensor_prod, axis=2)).T
        self.cur_tensor_prod = tensor_prod


    def gradient(self, W:np.ndarray, subset=None, subset_num=None):
        """
        Compute the gradient of the loss function

        Parameters:
            W (d,d) ndarray - The weights for the informative points
            subset (list) - A list of indices for the subset
            subset_num (int) - The number of the subset
        Returns:
            dW (d,d) ndarray - The gradient of the loss function
        """
        # Initialize the gradient
        self.update_gaussian(W, subset_num)
        dW = np.zeros((self.d,self.d))
        differences = self.differences if subset is None else self.subset_differences[subset_num]

        # If there is no subset, let the differences be the self.differeces and the one hot be the self.one_hot
        if subset is None:
            y_index = self.class_index
            one_hot = self.one_hot
        
        # Otherwise, select the subset
        else:
            y_index = []
            y_sub = self.y[subset]

            # Loop through the different classes and select the right subsets
            for i in range(self.num_classes):
                y_index.append(np.where(y_sub == self.classes[i])[0])

            # Modify the one hot encoding to just the subset
            one_hot = np.zeros((len(subset), self.num_classes))
            for i in range(len(subset)):
                one_hot[i, self.y_dict[y_sub[i]]] = 1

        # Loop through the different classes and select the right subsets
        for i in range(len(self.classes)):
            g_c = self.cur_gaussian[y_index[i]]
            g_c_totals = np.sum(g_c, axis=0)[:, np.newaxis, np.newaxis] + 1e-20
            product_c = self.cur_tensor_prod[y_index[i]]
            differences_c = differences[y_index[i]]

            # Calculate the weighted products
            weighted_product_c = g_c[:,:,np.newaxis] * product_c
            weighted_sum_c = np.sum(np.einsum('ijk,ijl->ijkl', weighted_product_c, differences_c), axis = 0)
            weighted_sum_c /= g_c_totals

            # Calculate the gradient first term
            dW += np.sum(one_hot[:,i][:,np.newaxis, np.newaxis] * weighted_sum_c, axis=0)

        # Calculate the gradient first term
        g_all_totals = np.sum(self.cur_gaussian, axis=0)[:, np.newaxis, np.newaxis] + 1e-20
        weighted_product_all = self.cur_gaussian[:,:,np.newaxis] * self.cur_tensor_prod
        weighted_sum_all = np.sum(np.einsum('ijk,ijl->ijkl', weighted_product_all, differences), axis = 0)
        weighted_sum_all /= g_all_totals

        # Calculate the gradient second term
        dW -= np.sum(weighted_sum_all, axis=0)

        # Return the regularized gradient
        return 2*dW + self.reg*(W / np.linalg.norm(W, 'fro') - self.dim_reg*np.eye(self.d) /self.d)
    

    def loss(self, W:np.ndarray, subset=None, subset_num=None):
        """
        Compute the loss function

        Parameters:
            W (d,d) ndarray - The weights for the informative points
            subset (list) - A list of indices for the subset
            subset_num (int) - The number of the subset
        Returns:
            loss (float) - The loss function
        """
        # Initialize the loss and total gaussian
        self.update_gaussian(W, subset_num)
        loss = 0
        total_g_log = np.log(np.sum(self.cur_gaussian, axis=0) + 1e-20)

        # Get the right y index
        if subset is None:
            y_index = self.class_index
        else:
            y_index = []
            for i in range(self.num_classes): # For each class
                y_index.append(np.where(self.y[subset] == self.classes[i])[0]) # Get the indices of the class within the subset
        
        # Loop through the different classes and select the right subsets
        for i in range(len(self.classes)):
            g_c = self.cur_gaussian[y_index[i]]
            loss += np.log(np.sum(g_c, axis=0) + 1e-20)

        # Calculate the loss
        return np.sum(loss - total_g_log) + self.reg*(np.linalg.norm(W, 'fro') - self.dim_reg*np.trace(W) /self.d)


    ########################## Optimization and Training Functions ############################
    def gradient_descent(self):
        """
        Perform gradient descent on the model
        Parameters:
            None
        Returns:
            None
        """
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

            # Check for convergence after a certain number of iterations
            break_value = np.linalg.norm(gradient)
            if break_value < self.tol*self.n and i > self.max_iter//50:
                break


    def stochastic_gradient_descent(self, re_randomize=True):
        """
        Perform stochastic gradient descent on the model

        Parameters:
            re_randomize (bool) - Whether to re-randomize the batches after each epoch
        Returns:
            None
        """
        # Raise an error if there are no epochs or batch size, or if batch size is greater than the number of points
        if self.batch_size is None or self.epochs is None:
            raise ValueError("Batch size or epochs must be specified")
        if self.batch_size > self.n:
            raise ValueError("Batch size must be less than the number of points")
        
        # Initialize the loop, get the batches, and go through the epochs
        batches = self.randomize_batches()
        loop = tqdm(total=self.epochs*len(batches), position=0)
        self.update_differences(self.X, batches)
        for epoch in range(self.epochs):

            # reset the batches if re_randomize is true
            if re_randomize and epoch > 0:
                batches = self.randomize_batches()
                self.update_differences(self.X, batches)
            
            # Loop through the different batches
            loss_list = []
            self.weights_history.append(self.weights.copy())
            for i, batch in enumerate(batches):

                # Get the gradient, update weights, and append the loss
                gradient = self.gradient(self.weights, subset = batch, subset_num = i)
                self.weights -= self.learning_rate * gradient
                loss_list.append(self.loss(self.weights, subset = batch, subset_num = i))

                # update our loop
                loop.set_description('epoch:{}, loss:{:.4f}'.format(epoch,loss_list[-1]))
                loop.update()

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
                # print(f"({epoch}) Val Accuracy: {np.round(val_accuracy,5)}.   Train Accuracy: {train_accuracy}")

            # Append the history of the weights
            self.weights_history.append(self.weights.copy())
            
        # Close the loop
        loop.close()


    def bfgs(self):
        """
        Perform Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimization on the model

        Parameters:
            None
        Returns:
            None
        """
        # Define iterations to show the progress and define the loss and gradient function in 1D
        show_iter = max(self.max_iter, 100) // 100
        loss_bfgs = lambda W: self.loss(W.reshape(self.d, self.d))
        gradient_bfgs = lambda W: self.gradient(W.reshape(self.d, self.d)).flatten()

        # Define the callback function
        def callback(weights):
            self.weights = weights.reshape(self.d, self.d)
            self.weights_history.append(self.weights.copy())

            # If there is a validation set, check the validation error
            if self.X_val_set is not None and self.y_val_set is not None:
                val_predictions = self.predict(self.X_val_set)
                val_accuracy = accuracy_score(self.y_val_set, val_predictions)
                self.val_history.append(val_accuracy)
                
                # Predict on the validation set and append the history
                train_predictions = self.predict(self.X)
                train_accuracy = accuracy_score(self.y, train_predictions)
                self.train_history.append(train_accuracy)

                # Show the progress
                if len(self.weights_history) % show_iter == 0:
                    print(f"({len(self.weights_history)}) Val Accuracy: {np.round(val_accuracy, 5)}.   Train Accuracy: {train_accuracy}")
        
        # Run the optimizer
        res = minimize(loss_bfgs, self.weights.flatten(), jac=gradient_bfgs, method='BFGS', options={'disp': False, 'maxiter': self.max_iter, 'gtol':self.tol}, callback=callback)
        self.weights = res.x.reshape(self.d, self.d)


    def fit(self, X:np.ndarray, y:np.ndarray, X_val_set=None, y_val_set=None, init_weights=None):
        """
        Fit the model to the data

        Parameters:
            X (n,d) ndarray - The data to fit the model on
            y (n,) ndarray - The labels of the data
            X_val_set (n_val,d) ndarray - The validation set for the data
            y_val_set (n_val,) ndarray - The validation labels for the data
            init_weights (d,d) ndarray - The initial weights for the model
        Returns:
            None
        """
        # Save the data as variables and encode y
        self.X = np.array(X)
        self.y = np.array(y)
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.encode_y(y)

        # Initialize the weights
        if init_weights is not None:
            self.weights = init_weights
        else:
            self.weights = .125*((np.random.random((self.d,self.d))*2 - 1) + np.eye(self.d))

        # If there is a validation set, save it
        if X_val_set is not None and y_val_set is not None:
            self.X_val_set = X_val_set
            self.y_val_set = y_val_set

        # Perform necessary differences calculations
        if self.optimizer != "sgd":
            self.update_differences(self.X)

        # Run the optimizer
        if self.optimizer == "sgd":
            self.stochastic_gradient_descent()
        elif self.optimizer == "bfgs":
            self.bfgs()
        elif self.optimizer == "grad":
            self.gradient_descent()

        # Otherwise, raise an error
        else:
            raise ValueError("Optimizer must be 'sgd', 'bfgs, or 'grad'")


    ############################## Prediction Functions #############################
    def predict(self, points:np.ndarray, show_probabilities=False):
        """
        Predict the labels of the data
        
        Parameters:
            points (n,d) ndarray - The data to predict
        Returns:
            predictions (n,) ndarray - The predicted labels of the data
        """
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


    ############################## Other Functions ###############################
    def copy(self):
        """
        Create a copy of the model

        Parameters:
            None
        Returns:
            new_model (model3 class) - A copy of the model
        """
        # Initialize a new model
        new_model = model3(max_iter=self.max_iter, tol=self.tol, learning_rate=self.learning_rate, 
                           reg=self.reg, dim_reg=self.dim_reg, 
                           optimizer=self.optimizer, batch_size=self.batch_size, epochs=self.epochs)
        
        # Save the attributes of the new model
        new_model.X = self.X
        new_model.y = self.y
        new_model.n = self.n
        new_model.d = self.d
        new_model.weights = self.weights
        new_model.classes = self.classes
        new_model.num_classes = self.num_classes
        new_model.y_dict = self.y_dict
        new_model.one_hot = self.one_hot
        new_model.differences = self.differences
        new_model.cur_gaussian = self.cur_gaussian
        new_model.cur_tensor_prod = self.cur_tensor_prod
        new_model.X_val_set = self.X_val_set
        new_model.y_val_set = self.y_val_set
        new_model.class_index = self.class_index
        new_model.val_history = self.val_history
        new_model.train_history = self.train_history
        new_model.weights_history = self.weights_history

        # Return the new model
        return new_model
    

    def save_weights(self, file_path):
        """
        Save the weights of the model to a file so that it can be loaded later

        Parameters:
            file_path (str) - The name of the file to save the weights to
        Returns:
            None
        """
        try:
            np.save(file_path, self.weights)
        except:
            raise ValueError("The file could not be saved")
        raise NotImplementedError("This function is not implemented yet")
    

    def load_weights(self, file_path):
        """
        Load the weights of the model from a file

        Parameters:
            file_path (str) - The name of the file to load the weights from
        Returns:
            None
        """
        try:
            self.weights = np.load(file_path)
        except:
            raise ValueError("The file could not be loaded")
        raise NotImplementedError("This function is not implemented yet")
        