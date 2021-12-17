'''
@author: Matt Facque

'''

import numpy as np
import math

#  preceptron_train(X,Y) function to build perceptron 
#  X = feature set
#  Y = label set
#  Output weights and bias
def perceptron_train(X,Y):
    #print(X)
    #print(Y)
    
    #  Create list of lists feature set with label set
    feature_set = X.tolist()
    label_set = Y.tolist()
    input = list(zip(feature_set, label_set))
    #print(input)
    
    #  Determine degree of feature set
    weight_count = len(feature_set[0])
    
    bias = 0
    weights = [0] * weight_count
    x_features = [0] * weight_count
    
    counter = 0
    
    while counter < 15:
        epoch = True
        for x in input:
            #temp_x1 = x[0][0]
            #temp_x2 = x[0][1]
            x_features = load_features(x[0])
            y = x[1]
            
            #  TESTING
            #print("Feature set = ", x_features)
            #print("label = ", y)
            #print("Weight set = ", weights)
            #print("bias = ", bias)
            
            #activation = (weights[0] * temp_x1) + (weights[1] * temp_x2) + bias
            activation = 0
            
            for i in range(len(weights)):
                #  TESTING
                #print("weight = ", weights[i])
                #print("x = ", x_features[i])
                activation += weights[i] * x_features[i]
                
            #print(activation)
            activation += bias   
            #print("activation = ", activation)
            
            total_activation = activation * y
            #  TESTING
            #print("y * activation = ", total_activation)
            
            if (total_activation <= 0):
                #weights[0] = update_weight(weights[0], temp_x1, y)
                #weights[1] = update_weight(weights[1], temp_x2, y)
                #bias = update_bias(bias, y)
                weights = update_W(x_features, weights, y)
                bias = update_B(bias, y)
                epoch = False
            
            #  TESTING
            #print("New feature\n******************")
            
        counter+=1
        #print(counter)
        
        #  If no change to weights or bias, end perceptron algorithm
        if (epoch):
            break
    
    return weights, bias
######                    

#  Make an array of features of n dimensions
#  Input is input[0] which is the sublist of features
def load_features(feature_set):
    features = []
    
    for x in feature_set:
        features.append(x)
        
    #print(features)
    
    return features
######

#  Update_weight function if activation <= 0 (mistake)
#  input is weight, feature value, label value
#  output is updated weight value
#def update_weight(w, x, y):
#    return (w + x * y)
######

#  Update bias function if activation <= 0 (mistake)
#  Input is bias value and label value
#  Output is updated bias value
#def update_bias(b, y):
#    return (b + y)
######

#  Update all weights using all feature values
#  Input is feature set, weight set
#  Note: x, w are lists
def update_W(x, w, y):
    count = len(w)

    for i in range(count):
        w[i] = w[i] + x[i] * y 
    
    #  TESTING
    #print("New weights = ", w)
    return w
######

#  Update bias using bias and label
#  Input is bias, label
def update_B(b, y):
    #  TESTING
    #print("bias = ", b + y)
    return (b + y)
######

#  Function to test perceptron model
#  Input is feature set and label set as well as trained weights and bias
#  Output is accuracy value for the model
def perceptron_test(X,Y,w,b):
    #  Create list of lists of feature set with label set
    feature_set = X.tolist()
    label_set = Y.tolist()
    input = list(zip(feature_set, label_set))
    
    accuracy = 0
    total = len(feature_set)
    
    for x in input:
        x_features = load_features(x[0])
        y = x[1]
        
        activation = 0
            
        for i in range(len(w)):
            activation += w[i] * x_features[i]
            
        activation += b
        
        if (activation < 0):
            activation = -1
        else:
            activation = 1
            
        if (activation == y):
            accuracy+=1
        else:
            accuracy+=0
    
    return accuracy / total
    ###  Go through test feature/label set,
    ###  w1 * x1 + w2 * x2 + b = #
    ### if (# < 0): # == -1
    ### else: # == 1
    ### compare label to #
    ### if (label == #): +1 to counter
    ### else: +0 to counter
    ### counter / len(input) = accuracy
    ### return accuracy
######    
    
    
    