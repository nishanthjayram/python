"""
    Implementation of the numerical integration method of adaptive quadrature through
    recursion. Additionally contains methods for other forms of numerical integration,
    as well as methods for differentiation.

    (Developed as part of a course on scientific computing.)

    Author  :  Nishanth Jayram (https://github.com/njayram44)
    Date    :   January 6, 2021
"""
import numpy as np
import matplotlib.pyplot as plt

### Integration methods
def int_tabulated_data(x, y, dbg=False):
    """Numerical integral of tabulated data.

    Parameters
    ----------
    x : numpy array
        a collection of x-values
    y : numpy array
        a collection of function values corresponding to x-values
    dbg : boolean
        if True, function returns an estimation of the integration accuracy
    
    Returns
    --------
    I : the integral of the tabulated data (calculated using the trapezoid rule)
    error : the estimation of the integration accuracy (returned only if dbg==True)
    """

    # Verify that inputs are non-empty numpy arrays.
    assert type(x) == np.ndarray
    assert type(y) == np.ndarray
    assert len(x) > 0
    assert len(y) > 0
    
    # Accumulate x_(i + 1) - x_i.
    p = x[0:len(x) - 1] - x[1:] # An array of x_(i + 1) - x_i

    # Accumulate f(x_i + 1) + f(x_i).
    q = (y[1:len(y)] + y[0:len(y)-1]) * (p ** 4) # An array of y_(i + 1) + y_i

    # Apply trapezoid rule. 
    I = (8*np.pi)/3 * np.sum((1/2) * p * q)

    # If true, then calculate error value and return with integral.
    if dbg==True:
        p_half = x[0:(len(x)-1)/2] - x[1:(len(x)/2)]
        q_half = (y[1:len(y)/2] + y[0:(len(y-1)/2)]) * (p ** 4)
        I_half = (8*np.pi)/3 * np.sum((1/2) * p_half * q_half)
        error = I - I_half
        return (I, error)
    
    return I

def int_riemann_sum(f, a, b, N=100):
    """Numerical integral by N-point Riemann sum."""
    x = a + np.sort(np.random.rand(N))*(b - a)
    dx = np.diff(x)
    I = sum(f(x[:-1])*dx)
    return I

def int_trapezoid(f, a, b, N=100):
    """Numerical integral by N-point trapezoid rule."""
    x = np.linspace(a, b, N)
    y = f(x)
    h = (b - a)/(N - 1)
    I = h*(0.5*y[0] + sum(y[1:-1]) + 0.5*y[-1])
    return I

def int_iterative_trapezoid(f, a, b, tol=1e-10, maxiter=30, dbg=False):
    """Numerical integral by iterative application of trapezoid rule."""
    I_old = int_trapezoid(f, a, b, N=2)
    for k in range(2,maxiter):
        N = 2**k
        I_new = int_trapezoid(f, a, b, N)
        err = np.abs(I_new - I_old)
        if err <= tol*I_old:
            break
        else:
            I_old = I_new
    if dbg:
        return (I_new, err, k)
    else:
        return I_new

# An implementation of adaptive quadrature.
def adaptive_integrator(f, a, b, N=4, tol=1e-10):
    Q_1 = int_trapezoid(f, a, b, N)
    Q_2 = int_trapezoid(f, a, b, N // 2)
    error = abs(Q_1 - Q_2)
    # If error greater than tolerance, then recurse and apply adaptive quadrature to
    # smaller subintervals.
    if error > tol:
        m = (a + b) / 2
        Q_2 = adaptive_integrator(f, a, m, N, tol) + adaptive_integrator(f, m, b, N, tol)
    return Q_2

### Differentiation methods
def deriv_1point_blind(f, x, h):
    """A one-point forward derivative approximation with fixed h."""
    return (f(x + h) - f(x))/h

def deriv_1point_optimal(f, x, eps_f=1e-15, xc=1):
    """A one-point forward derivative approximation with optimal h."""
    h = np.sqrt(eps_f)*xc
    return (f(x + h) - f(x))/h

def deriv_2point_blind(f, x, h):
    """A two-point symmetric drivative with fixed h."""
    return (f(x + h) - f(x - h))/(2*h)

def deriv_2point_optimal(f, x, eps_f=1e-15, xc=1):
    """A two-point symmetric derivative with optimal h."""
    h = eps_f**(1./3.)*xc
    return (f(x + h) - f(x - h))/(2*h)


### Analytically known functions to test things on
def ef(x):
    """A test function with known derivative."""
    return np.cos(x)

def efprime(x):
    """Derivative of ef."""
    return -np.sin(x)

def EF(x):
    """Antiderivative of ef."""
    return np.sin(x)


### Diagnostics helpers
def plot_deriv_residuals(deriv_method, **kwargs):
    """Use ef and efprime to visualize derivative residuals."""

    x = np.linspace(-np.pi, np.pi)
    yp_analytic = efprime(x)
    yp_numeric = deriv_method(ef, x, **kwargs)

    residuals = yp_numeric - yp_analytic
    print(f"Mean error = {np.mean(np.abs(residuals))}")
    plt.plot(x, residuals, '+', label=f"Mean error = {np.mean(np.abs(residuals))}")
    plt.xlabel('$x$')
    plt.ylabel(r"$df(x)_\mathrm{numerical}-df(x)_\mathrm{analytic}$")
    plt.legend()
    plt.show()