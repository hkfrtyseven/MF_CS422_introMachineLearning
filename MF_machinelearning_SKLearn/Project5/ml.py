'''
@author:         Matt Facque
@description:    Python module supporting machine learning models
@import:         sklearn --> ml models
'''
import numpy as np
import sklearn
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

'''
@Module:        model_test(X, model)
@Description:   Function to test the model of choice with testing data
@Parameters:    X = feature set (test)
                model = input model
@Output:        Set of predictions based on model and test feature set
'''
def model_test(X, model):
    #  Make predictions
    feature_Predict = model.predict(X)
    
    return feature_Predict
######

'''
@Module:        compute_F1(Y, Y_hat)
@Description:   Function to compute F1 which is a measure of accuracy of the model
@Parameters:    Y = labels obtained from predictions
                Y_hat = labels collected from testing data
@Output:        Value for F1
'''
def compute_F1(Y, Y_hat):
    f1_Score = f1_score(Y, Y_hat, average='macro')
    
    return f1_Score
######

'''
@Module:        dt_train(X, Y)
@Description:   Function to train a decision tree model using the features (X) and labels (Y)
@Parameters:    X = feature set
                Y = label set
@Output:        Trained decision tree model
'''
def dt_train(X, Y):
    #  Initialize decision tree
    dt = tree.DecisionTreeClassifier(random_state=0)
    #  Train decision tree 
    trained_dt = dt.fit(X,Y)
    
    return trained_dt
######

'''
@Module:        pca_train(X, K)
@Description:   Function to perform learning operation with PCA
@Parameters:    X = feature set
                K = Number of principal components to keep from data set
@Output:        A dataset with reduced dimensionality
'''
def pca_train(X, K):
    #  Apply dimensionality reduction
    principal_Comp = PCA(n_components=K, random_state=0)
    #  Feature set is reduced here
    reduced_Comps = principal_Comp.fit(X)
    
    return reduced_Comps
######

'''
@Module:        pca_transform(X, pca)
@Description:   Function to actually apply reduction to data
@Parameters:    X = feature set
                pca = dimensionality reduction
@Output:        Reduced dataset
'''
def pca_transform(X, pca):
    #  Apply transformation on X
    X_transformed = pca.transform(X)
    
    return X_transformed
######

'''
@Module:        kmeans_train(X)
@Description:   Function to train kmeans with the feature set
@Parameter:     X = feature set
@Output:        Trained cluster centers
'''
def kmeans_train(X):
    #  Produce clusters from training data
    trained_Kmeans = KMeans(random_state=0).fit(X)
    
    return trained_Kmeans
######

'''
@Module:        knn_train(X, Y, K)
@Description:   Function to train K nearest neighbors (KNN)
@Parameters:    X = feature set
                Y = label set
                K = value for nearest neighbors
@Output:        Trained nearest neighbors model
'''
def knn_train(X, Y, K):
    #  Set up nearest neigbors
    k_Neighbors = KNeighborsClassifier(K)
    
    kNeigh_fitted = k_Neighbors.fit(X,Y)
    
    return kNeigh_fitted
######

'''
@Module:        perceptron_train(X, Y)
@Description:   Function to train a perceptron model
@Parameters:    X = feature set
                Y = label set
@Output:        Trained perceptron model
'''
def perceptron_train(X, Y):
    #  Initialize perceptron
    perc = Perceptron(random_state=0)
    
    trained_Perc = perc.fit(X,Y)
    
    return trained_Perc
######

'''
@Module:        nn_train(X, Y, hls)
@Description:   Function to create and train a neural network
@Parameters:    X = feature set
                Y = label set
                hls = hidden layer size, number (size) of hidden layers in the nn
@Output:        Trained neural network
'''
def nn_train(X, Y, hls):
    #  Create the neural network
    nn = MLPClassifier(hidden_layer_sizes=hls, max_iter=1000, random_state=0)
    
    #  Train nn
    trained_Nn = nn.fit(X, Y)
    
    return trained_Nn
######

'''
@Module:        svm_train(X, Y, k)
@Description:   Function to train a support vector machine on the data
@Parameters:    X = training feature set
                Y = training label set
                k = kernal
@Output:        A fitted SVM
'''
def svm_train(X, Y, k):
    #  Create the SVM
    svm = SVC(kernel=k, random_state=0)
    
    #  Train the SVM
    trained_SVM = svm.fit(X, Y)
    
    return trained_SVM
######








