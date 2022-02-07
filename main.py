import numpy
import numpy as np

def act(x):
    return 0 if x <= 0 else 1


def log_or(operand1,operand2):
    x = np.array([operand1, operand2])
    w1 = np.array([0.5, 0.5])
    sum = np.dot(w1, x)
    print(sum,'сумма на нейроне')
    y = act(sum)
    return y


print(log_or(0,0))