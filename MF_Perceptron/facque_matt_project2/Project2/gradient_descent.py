'''
@author: Matt Facque
'''
import numpy as np

#  gradient_descent(grad_f, x_init, learning_rate) function to calculate gradient descent
#  Input:  grad_f, function to perform gradient descent on
#          x_init, initial value of x
#          learning_rate, alda(?), step size of gradient descent, learning rate
#  Output: Value < 0.00001
def gradient_descent(grad_f, X, learning_rate):
    #print(convergence)
    #print(x_init)
    #print(x_init[0])
    convergence = 0.0001
    
    x_val = X

    steps = 0
    
    #  while any of features > convergence and perform only 550 steps
    while np.any(abs(x_val) > convergence) and steps < 550:
        grad_X = grad_f(x_val)
        descent = grad_X * learning_rate
        x_val = x_val - descent
    
        steps+=1
        
        #if (steps == 52):
            #break
    #  Calculate gradient f(x) w/ x_init
    #grad_X = grad_f(X)
    # = 2*x[0] = [10.]
    #print(grad_X)
    #  Multiple by learning_rate
    #descent = grad_X * learning_rate
    # = 0.1 * [10.]
    #print(descent)
    #  X - descent
    #descended_X = X - descent
    # = [5.] - [1.]
    #print(descended_X)
    
    #print(steps)

    return x_val
######



