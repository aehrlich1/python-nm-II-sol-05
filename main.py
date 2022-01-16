import numpy as np
import matplotlib.pyplot as plt


# Question 01
def fixed_point_iteration_01(x0, i):
    x = np.empty(i)
    x[0] = x0
    for j in range(1, i):
        x[j] = 1 + (2 / x[j-1])

    return x


def fixed_point_iteration_02(x0, i):
    x = np.empty(i)
    x[0] = x0
    for j in range(1, i):
        x[j] = x[j-1]**2 - 2

    return x


def question_01():
    x1 = fixed_point_iteration_01(3, 10)
    x2 = fixed_point_iteration_02(3, 10)
    np.set_printoptions(precision=3)
    print("\nQuestion 1\n-------------------------------------------")
    print(f'{"x1":20} {"==>":15} {x1}')
    print(f'{"x2":20} {"==>":15} {x2}')

    _, ax = plt.subplots()
    ax.grid(linewidth=0.2)
    ax.set_xlabel("# Iterations")
    ax.set_ylabel("Sol")
    ax.plot(x1, 'o-', label='Test1', markersize=5)
    ax.legend()
    plt.show()


# Question 02
def bisection(fun, a, b):
    """
    Bisection methos

    Parameters
    ----------
    fun : function
        The function for which the root shall be found.
    
    a : int
        The lower bound of the Domain.

    b : int
        The upper bound of the Domain.
    
    Returns
    -------
    x : ndarray
        The scalar solution to the root.

    i : int
        The number of iterations required to achieve convergence.

    """
    tol = 1e-12
    i = 0

    while True:
        x = a + (b - a) / 2

        if(np.abs(fun(x)) < tol):
            return x, i
        elif(fun(a) * fun(x) > 0):
            a = x
        else:
            b = x

        i += 1


def fun(x):
    return x**3


def question_02():
    x, i = bisection(fun, -2, 1)
    print("\nQuestion 2\n-------------------------------------------")
    print(f'{"Solution":20} {"==>":15} {x}')
    print(f'{"Steps":20} {"==>":15} {i}')


# Question 03
def newton_method(fun, jacobi, x0):
    """
    Newton's methods in arbitrary dimension (n). Within each iteration
    the linear system J*d = -f(x) is solved for the update vector d.
    Convergence is achieved if in each dimension the absolute difference
    to the root is < tol.

    Parameters
    ----------
    fun : ndarray
        Function who's root we want to find. Returns an n-dimensional
        array.
        
    
    jacobi : n x ndarray
        The correspoding jacobi matrix of the function n.
        Returns an array of dimension n x n.
    
    Returns
    -------
    x : ndarray
        The solution vector of dimension n

    i : int
        The number of iterations required to achieve convergence.
        Returns 0 if no solution was found.

    """
    i = 1
    tol = 1e-3
    x = x0

    while True:
        d = np.linalg.solve(jacobi(x), -fun(x))
        x = x + d

        if(np.abs(fun(x).all()) < tol):
            return x, i

        if(i > 1e4):
            print("No convergence")
            return x, 0

        i += 1


def fun_2d(x):
    return np.array([x[0] * x[1] - 1, x[0]**3 - x[1]**3])


def jacobi(x):
    return np.array([[x[1], x[0]], [3*x[0]**2, -3*x[1]**2]])


def question_03():
    x0 = np.array([2, 3])
    x, i = newton_method(fun_2d, jacobi, x0)
    print("\nQuestion 3\n-------------------------------------------")
    print(f'{"Solution":20} {"==>":15} {x}')
    print(f'{"Steps":20} {"==>":15} {i}')


# Question 04
# Main
question_01()
question_02()
question_03()
