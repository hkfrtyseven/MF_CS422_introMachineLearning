'''
@author:        Matt Facque
@project:       Project 4
@description:   PCA algorithm
'''

import numpy as np

'''
 @module:         compute_Z(X, centering=True, scaling=False)
 @description:    Center and scale the input matrix X
 @parameters:     X = input matrix
                  centering = boolean, center X
                  scaling = boolean, scale X
'''
def compute_Z(X, centering=True, scaling=False):
    #  Cast input array to float
    X_float = X.astype(float)
    #  Dimension of numpy array, dim = tuple 
    dim = X.shape
    #print(dim)
    #  Number of total samples
    total_X = dim[0]
    #print(total_X)
    #  Number of features
    size = dim[1]
    #print(size)
    #  numpy array of mean values, initialized as zeroes
    means = np.zeros(size)
    #print(means)
    #  numpy array of standard deviation values, initialized as zeroes
    scaling_factors = np.zeros(size)
    #print(scaling_factors)
    
    #print(X)

#    if (centering == True):
#        for i in range(size):
#            count = 0
#            for j in X:
#                arr_val = X[j][i]
#                print(arr_val)
#                count += arr_val
#            means[i] = count / total_X
#        for i in range(size):
#            for j in X:
#                X[j][i] -= means[i]
#if (scaling == True):
#        for i in range(size):
#            count = 0
#            for j in X:
#                count += X[j][i]
#            total_count = count / total_X
#            scaling_factors[i] = math.sqrt(total_count)
#        for i in range(size):
#            for j in X:
#                X[j][i] = X[j][i] / scaling_factors[i]

    if (centering == True):
        for i in range(total_X):
            means += X_float[i]
        means = means / total_X
        #print(means)
        for i in range(total_X):
            X_float[i] -= means
                   
    if (scaling == True):
        for i in range(total_X):
            scaling_factors += np.square(X_float[i] - means)
        scaling_factors = scaling_factors / total_X
        #print(scaling_factors)
        scaling_factors = np.sqrt(scaling_factors)
        for i in range(total_X):
            X_float[i] = X_float[i] / scaling_factors
                
    #print(X_float)
    
    return X_float
######

'''
 @module:        compute_covariance_matrix(Z)
 @description:   Output COV from input, Z(transposed) * Z
 @parameter:     Z, input matrix as centered and scaled numpy array
'''
def compute_covariance_matrix(Z):
    #print(Z)
    #  Transpose Z
    transpose_Z = np.transpose(Z)
    #print(transpose_Z)
    #  Covariance matrix
    COV = np.matmul(transpose_Z, Z)
    #print(COV)
    
    return COV
######

'''
 @module:       find_pcs(COV)
 @description:  Input covariance matrix and output a matrix of eigenvalues and eigenvectors
                The eigenvectors will be ordered largest to smallest and each column will 
                be a singular eigenvector
 @parameter:    COV, covariance matrix
 @output:       eig_Val = numpy array of eigenvalues
                eig_Vec = numpy array of eigenvectors, each column is an eigenvector
'''
def find_pcs(COV):
    #print(COV)
    
    eig_Val, eig_Vec = np.linalg.eig(COV)
    #print(eig_Val)
    #print('\n')
    #print(eig_Vec)
    #transpose_eigVec = np.transpose(eig_Vec)
    #print(transpose_eigVec)
    
    return eig_Val, eig_Vec
######

'''
 @module:        project_data(Z, PCS, L, k, var)
 @description:   Transform the standardized data set Z into the projected data set Z*
                 using the eigenvectors and eigenvalues.
 @parameter:     Z = standardized data set
                 PCS = array of eigenvectors
                 L = array of eigenvalues
                 k = number of principal components used to project the data, 0 <= k <= D
                 var = value of the desired cumulative variance produced by the projection, 0 <= var <= 1
'''
def project_data(Z, PCS, L, k, var):
    if k > 0:
        #  Choose number of PCS 
        k_Eigenvectors = np.zeros(k)
        k_Eigenvectors = np.copy(PCS[:k])
            
    else:
        #  Total all eigenvalues
        count = np.sum(L)
        #print(count)
        #  Turn eigenvalues into corresponding percentages
        L = L / count
        #print(L)
        eig_Range = 0
        for x in range(np.size(L)):
            #print('loop number: ', x)
            if L[x] >= var:
                #print(L[x])
                eig_Range = x
                break
            else:
                #print(L[x+1])
                L[x+1] += L[x]
                #print(L)
        #print(eig_Range)
        k_Eigenvectors = np.zeros(eig_Range)
        k_Eigenvectors = np.copy(PCS[:eig_Range])
    
    #Z_size = Z.shape
    #print(Z_size)
    #u_Size = k_Eigenvectors.shape
    #print(u_Size)
    transpose_k = np.transpose(k_Eigenvectors)
    #print(transpose_k)
    #print(transpose_k.shape)
    Z_Star = np.matmul(Z, transpose_k)
    #print(Z_Star)        
    return Z_Star
        