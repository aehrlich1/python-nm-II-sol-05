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

    _, ax = plt.subplots()
    ax.grid(linewidth=0.2)
    ax.set_xlabel("# Iterations")
    ax.set_ylabel("Sol")
    ax.plot(x1, 'o-', label='Test1', markersize=5)
    ax.legend()
    plt.show()


# Question 02

# Question 03

# Question 04

# Main
question_01()
