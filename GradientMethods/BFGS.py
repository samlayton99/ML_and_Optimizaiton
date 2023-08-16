import numpy as np
from numpy.linalg import inv, norm

def BFGS(f,df,x0,A0,max_iter=40,tol=1e-5):
    "Make the BGFS algorithm to find the minimizer of f"
    # initialize the stopping criteria, iteration counter, and the inverse Hessian
    done = False
    iters = 1
    A_inv = inv(A0)

    # initialize the first step
    x = x0 - A_inv @ df(x0)
    s = x - x0

    # loop until the stopping criteria is met and increment the iteration counter
    while not done:
        iters += 1

        # update our y value and make helpful vectors/variables
        y = df(x) - df(x0)
        sy = s @ y
        Ay = A_inv @ y

        # Calculate the new inverse Hessian approximation and update the x value
        A_inv = (A_inv + ((sy + y @ Ay)/sy**2) * np.outer(s,s) - (np.outer(Ay,s) + np.outer(s,Ay))/sy)
        x0 = x
        x = x0 - A_inv @ df(x0)

        # stopping criteria and update the step
        s = x - x0
        done = ((norm(s) < tol) or (norm(df(x)) < tol) or (np.abs(f(x) - f(x0)) < tol) or (iters >= max_iter))
    
    # return the minimizer and the number of iterations
    if iters >= max_iter:
        iters = float("nan")
    return x, iters

# Define our functions and derivative
f = lambda x: x[0]**3 - 3*x[0]**2 + x[1]**2
df = lambda x: np.array([3*x[0]**2 - 6*x[0], 2*x[1]])

# Define our initial guess and Hessian approximation arrays that we will loop through
points =[np.array([4,4]), np.array([4,4]), np.array([10,10]), np.array([10,10])]
hessian = [np.array([[18,0],[0,2]]),np.eye(2), np.array([[54,0],[0,2]]),np.eye(2)]

# Loop through each of the initial conditions and run the BFGS algorithm
for i in range(len(points)):
    x = BFGS(f,df,points[i],hessian[i])
    
    # Print the name of the question, the minimizer, and the number of iterations
    print("Part", str(i+1)+":")
    print("The minimizer is: ", x[0])
    print("The number of iterations is: ", x[1], "\n")

# Part 5:
x = BFGS(f,df,np.array([0,0]),hessian[0])
print("\nPart 5:")
print("The minimizer is: ", x[0])
print("The number of iterations is: ", x[1], "\n")
print("We see clearly that if we start at the origin, the algorithm does not converge.\nThis is because Df(x0) is 0, hence x1 = x0, and so on. This makes y_k = 0,\nthus causing nan values. In reality, x stays unchanged, as each iteration stays at 0.\nZero is already a local minimum, so the algorithm cannot work.\n")