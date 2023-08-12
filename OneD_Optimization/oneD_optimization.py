# oneD_optimization.py
"""Volume 2: One-Dimensional Optimization.
<Sam Layton>
<001>
<2/2/23>
"""
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def golden_section(f, a, b, tol=1e-5, maxiter=100):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # Initialize x0 and the golden ratio, and set the converged to False.
    x0 = (a + b)/2.0
    gold = (1 + 5**.5)/2
    converged = False

    # Iterate until the stopping criterion is met.
    for i in range(maxiter):

        # Compute the new points a1, and b1, and c
        c = (b-a)/gold
        a1 = b - c
        b1 = a + c

        # If f(a1) < f(b1), then b = b1, else a = a1.
        if f(a1) < f(b1):
            b = b1
        else:
            a = a1

        # Compute the new x1, and check if the stopping criterion is met.
        x1 = (a + b)/2.0
        if abs(x0 - x1) < tol:
            converged = True
            break

        # Update x0
        x0 = x1

    # Return the minimizer, convergence, and the number of iterations.
    return x1 , converged, i + 1



# Problem 2
def newton1d(df, d2f, x0, tol=1e-5, maxiter=100):
    """Use Newton's method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # Initialize the converged variable to False and iterate through the max iterations.
    converged = False
    for i in range(maxiter):

        # Compute the new x1, and check if the stopping criterion is met.
        x1 = x0 - df(x0)/d2f(x0)
        if abs(x1 - x0) < tol:
            converged = True
            break

        # Update x0. When the stopping criterion is met, x1 is returned.
        x0 = x1
    return x1, converged, i + 1


# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiter=100):
    """Use the secant method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        x1 (float): Another guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # Initialize the converged variable to False and get first derivative.
    converged = False
    dfx0 = df(x0)

    # Iterate through the max iterations and get derivative of x1, and hold x0.
    for i in range(maxiter):
        dfx1 = df(x1)
        hold = x1

        # Compute the new x1, and update x0 and dfx0.
        x1 = (x0 * dfx1 - x1 * dfx0) / (dfx1 - dfx0)
        x0 = hold
        dfx0 = dfx1

        # Check if the stopping criterion is met.
        if abs(x1 - x0) < tol:
            converged = True
            break

    # Return the minimizer, convergence, and the number of iterations.
    return x1, converged, i + 1


# Problem 4
def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """
    # Get the derivative of f at x in direction p, and compute the function value at x.
    Dfp = np.dot(Df(x), p)
    fx = f(x)

    # Iterate until the Armijo condition is met.
    while f(x + alpha * p) > fx + c * alpha * Dfp:
        alpha *= rho

    # Return the optimal step size
    return alpha



# test cases
def testProb1():
    """Test the golden section search algorithm on the function."""
    f = lambda x: np.exp(x) - 4*x
    a = 0
    b = 3
    x, converged, iters = golden_section(f, a, b)
    print("x =", x)
    print("converged =", converged)
    print("iterations =", iters)

def testProb2():
    """Test Newton's method on the function."""
    df = lambda x: 2*x + 5*np.cos(5*x)
    d2f = lambda x: 2 - 25*np.sin(5*x)
    x0 = 0
    x, converged, iters = newton1d(df, d2f, x0)
    print("x =", x)
    print("converged =", converged)
    print("iterations =", iters)

def testProb3():
    """Test the secant method on the function."""
    f = lambda x: x **2 + np.sin(x) + np.sin(10*x)
    df = lambda x: 2*x + np.cos(x) + 10 * np.cos(10*x)
    x0 = 0
    x1 = -1
    x, converged, iters = secant1d(df, x0, x1)
    print("x =", x)
    print("converged =", converged)
    print("iterations =", iters)

    # plot the function and the minimizer
    domain = np.linspace(-2, 2, 100)
    plt.title("Secant Method Minimizer")
    plt.plot(domain, f(domain))
    plt.plot(x, f(x), 'ro')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.show()

def testProb4():
    f = lambda x: x[0]**2 + x[1]**2 + x[2]**2
    Df = lambda x: np.array([2*x[0], 2*x[1], 2*x[2]])
    x = np.array([1, 0, .2])
    p = np.array([-1, 1, -1])
    alpha = backtracking(f, Df, x, p)
    print("alpha =", alpha)


#testProb1()
#testProb2()
#testProb3()
#testProb4()
