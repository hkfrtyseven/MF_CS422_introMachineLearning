'''
@author: Matt Facque
'''

import scipy.spatial as sc
import numpy as np

def load_data(fname):
    f = open(fname, 'r')
    ctr = 0
    y_str = ''
    for line in f:
        line = line.strip().split(';')
        if ctr == 0:
            x_str = line
        else:
            y_str = line
        ctr+=1
    f.close()
    X = []
    Y = []
    for item in x_str:
        temp = [float(x) for x in item.split(',')]
        X.append(temp)
    if len(y_str)>0:
        for item in y_str:
            temp = int(item)
            Y.append(temp)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

#  KNN_test(X_train, Y_train, X_test, Y_test, K)
#  X_train = training data set
#  Y_train = training label set
#  X_test = testing data set
#  Y_test = testing label set
#  K = value of nearest neighbors
def KNN_test(X_train, Y_train, X_test, Y_test, K):
    K_neighbors = K
    min_val = 999999
    distance_array = []
    total_y = 0
    acc = 0
    counter = 0
    np.set_printoptions(precision = 2, linewidth = 200)
    distance_array = sc.distance.cdist(X_train, X_test, 'euclidean')
    #print(distance_array)
    # Algorithm:
    # iterate through distance array (nested for loops)
    for i in distance_array:
        for j in i:
            counter = 0
            if (j < min_val):
                min_val = j
                x_TrainNN = X_train[counter]
                y_TrainNN = Y_train[counter]
                y_TestNN = Y_test[counter]
            counter+=1    
        if (y_TrainNN == y_TestNN):
            acc+=1
        else:
            acc+=0
    
    total_Y = len(Y_train)
    total_acc = acc / total_Y
    
    return total_acc
    #return 0
######  KNN_Test

#X,Y = load_data("data_4.txt")
#print(X)
#print(Y)
#acc = KNN_test(X,Y,X,Y,1)
#print(acc)
