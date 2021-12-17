README for Project2 by Matt Facque

perceptron.py
The perceptron algorithm simply traverses through the data set
calculating the activation with each feature (x1, x2,....,xn) and 
each weight (w1, w2,.....,wn).  Then the activation is multiplied with
the lable value which determines if the weights and bias need to be updated.
After a set number of iterations or if the algorithm goes through each data point
and doesn't update weights and bias', the algorithm finishes and returns the trained weights and bias.

perceptron_train(X,Y)
Perceptron_train is the function output the correct weights and bias.  The input is X - feature set and
Y - label set.  The function combines the feature set and label set using the zip() function, turning the 
two lists into a tuple.  Then the tuple processed with the features for the data point in x[0] (as a sublist)
the label for the data point in x[1].  Activation is calculated using the feature set and weight set and the
variable total_activation is the learning value, label * activation, which defines whether or not the algorithm 
needs to update weights and bias'.

load_features(x[0])
Helper function to put all the feature values of the data set into a standalone list.  This is important when working
with n-dimensional data.  After this function, the feature set is a standalone object and the label is a standalone object.

update_W(x_features, weights, y)
Helper function to update weights.  x_features is the current feature set as a list, weights is the current weight set also a list, 
and the data point's label value, y, is an integer.  The function traverses through the weight set, updating each weight, one at a time.  
The updated weight set, as a list, is returned from the function.

update_B(bias, y)
Helper function to update bias.  bias is the current bias value and y is the data set's label value.  The simple function returns the 
value gained from adding the current bias to the data point's label value.

perceptron_test(X,Y,w,b)
Perceptron_test is the function to test a data set using the weights and bias calculated from perceptron_train.  X is the data set to be test, 
Y is the label set for the data set, w is trained weight values and b is the trained bias value.  The perceptron algorithm for testing simply 
calculates the activation using the feature values (x1, x2,....,xn) and the weight values (w1, w2,....,wn), then the algorithm multiplies 
the activation by the label of the data set.  The value gained from that is compared to the label value of the data set and if the two numbers 
have the same sign, that is considered a success.  The accuracy of the model is defined as the number of success divided by the total number of 
data points.

gradient_descent.py
The gradient descent algorithm uses the gradient of a function to make "steps" toward the minimum of the function.  The algorithm is relatively simple,
the gradient of the functio you wish to determine the minimum of is calculated, in our case x^2 becomes 2x and x^3 becomes 3x^2.  The gradient is calculated
with an initial value of x and then multipled by the learning rate.  The learning rate determines the rate at which you "descend" the gradient of the function.
The learning rate multipled by the gradient is then subtracted from the previous value of x (the algorithm generally runs in a loop).  The algorithm runs to 
completion with the value of the function is less than a very small value (0.0001).  This value represents the minimum of the function.  The goal of the algorithm 
is to minimize your function of choice for some reason (determine the smallest value of x that will lead to an event happening, etc.).

My algorithm is relatively simple.  It will loop through the calculations until a set number of "steps" is achieved or if the value of x is determined to be
less than the target minimum value.  The return of the function gradient_descent is the minimal value of the function. 