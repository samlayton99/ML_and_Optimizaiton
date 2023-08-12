# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
<Name>
<Class>
<Date>
"""
import numpy as np
import scipy.optimize as opt
from scipy import linalg as la
from matplotlib import pyplot as plt

# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # Initialize convergence flag and loop through iterations.
    converged = False
    for i in range(maxiter):
        
        # find your optimal step size using the opt.minimize_scalar function and then update x
        alpha = opt.minimize_scalar(lambda alpha: f(x0 - alpha * Df(x0))).x
        x = x0 - alpha * Df(x0)

        # check for convergence and break if necessary
        if np.linalg.norm(Df(x), np.inf) < tol:
            converged = True
            break

        # update x0 and then return the minimizer, convergence flag, and number of iterations when appropriate
        x0 = x
    return x, converged, i + 1


# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Define our first residual and direction and our counter
    r0 = Q@x0 - b
    d0 = -r0
    k = 0

    # Initialize our variables that will be iterated on
    d = d0
    r = r0
    x=x0
    
    # Loop through until the norm of the residual is less than the tolerance or we have reached the maximum number of iterations
    while np.linalg.norm(r, np.inf) >= tol and k < len(b):
        # Define our alpha and calculate our next x and residual
        alpha = r@r / (d@(Q@d))
        x = x0 + alpha*d0
        r = r0 + alpha*Q@d0

        # Calculate our beta and direction, then update our counter
        beta = r@r / (r0@r0)
        d = -r + beta*d0
        k += 1

        # Update our variables
        r0 = r
        d0 = d
        x0 = x
    
    # Return the minimizer, convergence, and number of iterations
    return x, np.linalg.norm(r, np.inf) < tol, k


# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # Initialize our recurring and initial variables, and our counter
    r0 = -df(x0)
    d0 = r0
    r = -df(x0)
    k = 0
    
    # Loop through until the norm of the residual is less than the tolerance or we have reached the maximum number of iterations
    while np.linalg.norm(r, np.inf) >= tol and k < maxiter:

        # Calculate our new r, beta, and new direction vectors 
        r = -df(x0)
        beta = r@r / (r0@r0)
        d = r + beta*d0

        # calculate our new alpha, update our x, and increment our counter
        alpha = opt.minimize_scalar(lambda a: f(x0 + a*d)).x
        x = x0 + alpha*d
        k += 1

        # Update our variables
        r0 = r
        d0 = d
        x0 = x

    # Return the minimizer, convergence, and number of iterations
    return x, np.linalg.norm(r, np.inf) < tol, k
    

# Problem 4
def prob4(filename="linregression.txt",
          x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    # Load the data
    data = np.loadtxt(filename)

    # get the output and replace the first column with ones
    b = data[:, 0].copy()
    data[:, 0] = 1

    # Get the variables for the normal equation
    A = data.T @ data
    b = data.T @ b

    # Solve the linear system using conjugate gradient and return it
    x, converged, iterations = conjugate_gradient(A, b, x0)
    return x


# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        # use opt.fmin_cfg() to calculate the optimal beta values
        minf = lambda beta: np.sum([np.log(1+np.exp(-(beta[0] + beta[1]*x)))+(1-y)*(beta[0]+beta[1]*x)])
        self.b0 = opt.fmin_cg(minf, guess)[0]
        self.b1 = opt.fmin_cg(minf, guess)[1]

    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        # Calculate the probability of the outcome being 1
        return 1/(1+np.exp(-(self.b0 + self.b1*x)))


# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    # Load the data and fit the logistic regression
    data = np.load(filename)
    sigmoid = LogisticRegression1D()
    sigmoid.fit(data[:, 0], data[:, 1], guess)

    # Plot the data and the logistic curve
    x = np.linspace(30, 100, 1000)
    plt.plot(x, sigmoid.predict(x), label = "Logistic Curve", color = "orange")
    plt.scatter(data[:, 0], data[:, 1], label = "Data")
    plt.plot(31, sigmoid.predict(31), 'ro', label = "Failed Launch")

    # Set the labels and a title
    plt.title("Probability of O-Ring Failure")
    plt.xlabel("Temperature (F)")
    plt.ylabel("Probability of Failure")

    # Show the legend and plot, and return the probability of failure
    plt.legend()
    plt.show()
    return sigmoid.predict(31)


def test1():
    f = lambda x: x[0]**4 + x[1]**4 + x[2]**4
    Df = lambda x: np.array([4*x[0]**3, 4*x[1]**3, 4*x[2]**3])
    x0 = np.array([100, 0, 1000])
    print(steepest_descent(f, Df, x0))


def test2():
    n = 4
    A = np.random.random((n,n))
    Q = A.T @ A
    b, x0 = np.random.random((2,n))
    x = conjugate_gradient(Q, b, x0)
    print(np.allclose(Q @ x[0], b))
    print(la.solve(Q, b))
    print(x[0])
    print(x[2])

def test3():
    start = np.array([10,10])
    print(opt.fmin_cg(opt.rosen, start, fprime=opt.rosen_der))
    # rosenbrock f
    f = lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    df = lambda x: np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
    print(nonlinear_conjugate_gradient(f, df, start, maxiter = 250))