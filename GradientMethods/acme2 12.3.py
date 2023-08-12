from autograd import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from autograd import grad


# -------------------------------- Problem 12.15 ----------------------------------
def quad_grad_descent(Q, b, x0, eps=1e-6):
    """Minimize the quadratic function f(x) = (1/2)x^TQx + b^Tx + c, using Gradient Descent."""
    # Check if x0, b are column vectors of length n, and Q is an n x n matrix
    n = len(x0)
    assert len(b) == n
    assert Q.shape == (n,n)
    maxiter = 1000

    # Loop through iterations until convergence
    for i in range(maxiter):
        # Find our descent direction and get its size
        d = Q @ x0 + b
        size = np.linalg.norm(d)

        # Calculate the step size and calculate the next x
        a = size**2 / (d.T @ Q @ d)
        x1 = x0 - a * d
        x0 = x1

        # Check if we have converged and assign x0 to x1
        if size < eps:
            break
    
    # Return the final x
    return x1


# -------------------------------- Problem 12.16 ----------------------------------


# Now give the newton's method for the quadratic function
def exact_grad_descent(f, x0, eps=1e-5, maxiter=1000):
    # Give the derivative of f and start the counter
    df = grad(f)
    df2 = grad(f)
    count = 0

    # Loop through iterations until convergence or max iterations
    for i in range(maxiter):
        d = df(x0)
        size = np.linalg.norm(d)
        
        # Define our function phi and find the step size
        phi = lambda a: f(x0 - a*d)
        a = optimize.minimize_scalar(phi).x
        
        # Calculate the next x and assign x0 to x1
        x1 = x0 - a * d
        x0 = x1
        count += 1

        # Check if we have converged and break if so
        if size < eps:
            break

    # Return the final x
    return x1, count



# -------------------------------- Problem 12.17 ----------------------------------

# Define the function and the initial point, then run the algorithm
f = lambda x: 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
x0 = np.array([-2.0,2.0])
point, itters = exact_grad_descent(f, x0, maxiter=20000)

# print the point and number of iterations
print("The point is: ", point)
print("The number of iterations is: ", itters)