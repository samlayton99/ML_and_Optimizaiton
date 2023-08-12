import numpy as np
from scipy import linalg as la


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        matrix ((n,n) ndarray): the transition matrix for the Markov chain.
        labels (list(str)): a list of n labels corresponding to the n states.
        states (dict(str:int)): a dictionary mapping state labels to their
            indices in the transition matrix.
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        # Convert A to a numpy array if it isn't already, and get its shape
        A= np.array(A)
        m,n = A.shape

        # Check that A is square
        if m != n:
            raise ValueError("A is not square")

        # Check that A is column stochastic
        elif not np.allclose(np.sum(A, axis=0), np.ones(n)):
            raise ValueError("A is not column stochastic")
        
        # If no state labels are provided, use the indices 0, 1, ..., n-1
        if states is None:
            states = range(n)

        # Check that the number of state labels matches the dimension of A
        elif len(states) != n:
            raise ValueError("The number of state labels does not match A")

        # Save the state labels to a dictionary mapping labels to indices
        self.states = {states[i]:i for i in range(n)}

        # Save the transition matrix and the state labels
        self.matrix = A
        self.labels = states

        


    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        # Get the index of the current state and access the column of the matrix
        probabilities = np.zeros(len(self.labels))
        
        weight = 1
        index = [self.states[key] for key in state]
        
        for i in index:
            probabilities += self.matrix[:,i] * weight
            weight = (weight + 1)**2
        
        probabilities = probabilities/np.sum(probabilities)
        
        # Make a random draw from the probabilities and return the state label
        j = np.argmax(np.random.multinomial(1, probabilities))
        return self.labels[j]
        

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        # Raise a value error if start is not in the state labels
        if start not in self.labels:
            raise ValueError("start must be in the state labels")

        # Initialize the list of states with the starting state
        state_list = [start]

        # Use the transition() method to transition from state to state N-1 times, appending the state label at each step
        for i in range(N-1):
            start = self.transition(start)
            state_list.append(start)
        
        # Return the list of states
        return state_list


    # Problem 3
    def path(self, start, stop,k=1):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        # Raise a value error if start and stop are not in the state labels
        if start not in self.labels or stop not in self.labels:
            raise ValueError("start and stop must be in the state labels")

        # Initialize the list of states with the starting state
        state_list = [start]
        new = start
        input = [start for i in range(k)]

        # Use the transition() method to transition from state to state N-1 times, appending the state label at each step
        while new != stop:
            new = self.transition(input)
            state_list.append(new)
            input.remove(input[0])
            input.append(new)
        
        # Return the list of states
        return state_list



    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        # Generate a random steady state vector and make it left stochastic
        x = np.array([np.random.random() for i in range(len(self.states))])
        x = x / np.sum(x)

        # Iterate until the steady state vector converges, multiplying by A each time
        for i in range(maxiter):
            x_new = np.dot(self.matrix, x)

            # Check for convergence
            if np.linalg.norm(x_new - x) < tol:
                return x_new

            # Update the steady state vector
            x = x_new

        # Raise a value error if there is no convergence within maxiter iterations
        string = "There is no convergence within " + str(maxiter) + " iterations"
        raise ValueError(string)


class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        inherited from MarkovChain:
            matrix ((n,n) ndarray): the transition matrix for a Markov chain with n states.
            labels (list(str)): a list of n labels corresponding to the n states.
            states (dict(str,int)): a dictionary mapping a state's label to its index.
        

    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        # Read the file and split it into lines
        with open(filename, "r") as f:
            lines = f.readlines()
        
        # Initialize words and sentences
        words = ["$tart"]
        sentences = []

        # Iterate through the lines, splitting each line into words and appending them to words
        for line in lines:
            newlines = line.split()
            
            # Go through each word and add it to words
            for word in newlines:
                # Add the word to the list of words if it is not already there (unique)
                if word not in words:
                    words.append(word)

            # Append $tart and $top to the beginning and end of the line, respectively and append it
            newlines.append("$top")
            newlines.insert(0, "$tart")
            sentences.append(newlines)

        # Append $top to the list of words
        words.append("$top")
        
        # Initialize the transition matrix
        matrix = np.zeros((len(words), len(words)))

        # Iterate through the sentences, adding 1 to the matrix for each transition
        for sentence in sentences:
            for i in range(len(sentence)-1):
                # Get the indices of the current and next words
                row_pos = words.index(sentence[i+1])
                col_pos = words.index(sentence[i])

                # Add 1 to the matrix for each transition
                matrix[row_pos,col_pos] += 1

        # Add a 1 to the $top entry of the matrix
        matrix[len(words)-1, len(words)-1] = 1
        
        # Make it left stochastic and initialize the MarkovChain
        matrix = matrix / np.sum(matrix, axis=0)
        MarkovChain.__init__(self, matrix, words)
        

    # Problem 6
    def babble(self, k = 5):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        # Get the path from $tart to $top
        path = self.path("$tart", "$top",k)

        # Remove the $tart and $top labels from the path
        path.remove("$tart")
        path.remove("$top")

        # Return the path as a string
        return " ".join(path)

chain = SentenceGenerator("firstNephi.txt")
print(chain.babble())
