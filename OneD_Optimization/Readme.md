# One Dimensional Optimization

One-dimensional minimization forms the basis of many advanced numerical techniques seen across various scientific and engineering domains. This project comprises classical algorithms for this minimization, serving as foundational tools in optimization theory.

## Project Flow

1. **Initialization**: Start by setting up and defining the primary functions integral to the various minimization algorithms.
2. **Golden Section Search**: Use the `golden_section` function to determine the approximate minimizer of the specified function.
3. **Newton's Method**: Engage the `newton1d` function which leverages both the first and second derivatives to zero in on the minimizer of the function.
4. **Secant Method**: With two initial points in hand, deploy the `secant1d` function to hone in on the function's minimizer.
5. **Backtracking Line Search**: The `backtracking` function stands ready to pinpoint the optimal step size for any function at a designated point, guaranteeing a sufficient decrease in the function.
6. **Testing Phase**: Use the test functions: `testProb1`, `testProb2`, `testProb3`, and `testProb4` to gauge the efficacy of the previously mentioned algorithms.

## Mathematical Background and Overview of Functions

### Golden Section Search

- **Theory**: The method, rooted in the golden ratio (approximately \( \phi = \frac{1 + \sqrt{5}}{2} \)), is an iterative procedure aimed at uncovering the local minimum of a function.
  
- **Function**: `golden_section(f, a, b, tol=1e-5, maxiter=100)`

### Newton's Method for One-Dimensional Minimization

- **Theory**: Operating as a root-finding algorithm, it utilizes both the first and second derivatives to determine the function's minimum.
  
- **Function**: `newton1d(df, d2f, x0, tol=1e-5, maxiter=100)`

### Secant Method for One-Dimensional Minimization

- **Theory**: Resembling Newton's method but only necessitating the first derivative, it starts with two initial points and iteratively zeros in on the minimum.
  
- **Function**: `secant1d(df, x0, x1, tol=1e-5, maxiter=100)`

### Backtracking Line Search with Armijo Condition

- **Theory**: It's an optimization tool designed to identify a fitting step size, ensuring a significant decrease in the function with each step.
  
- **Function**: `backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4)`


## How to Use

1. Import requisite functions from this module.
2. Establish or furnish the correct function, its derivative, the initial guess, and other parameters fitting the function you plan to utilize.
3. Invoke the function and decipher the outcomes.

## Dependencies

- numpy
- matplotlib

## Additional Information

For an evaluation of the functions, simply un-comment the associated `testProbX()` calls at the script's conclusion. This step offers a clear window into the functioning and proficiency of each algorithm, backed by distinct test cases.
