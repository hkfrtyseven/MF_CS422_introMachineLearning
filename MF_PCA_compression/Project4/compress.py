'''
@author:         Matt Facque
@project:        Project4
@description:    Image compression
'''

import numpy as np
import matplotlib.pyplot as plt
import pca
import os

#  GLOBAL
dimensions = (60,48)

'''
 @module:         load_data(input_dir)
 @description:    Load image data into matrix
 @parameters:     input_dir = directory containing images for compression
 @output:         DATA matrix as numpy array
                  Columns = samples
                  Rows = features
'''
def load_data(input_dir):
    #images = len(os.listdir(input_dir))
    #print(images)
    data = np.zeros(shape=(2880,))
    
    for x in os.listdir(input_dir):
        #print(filename + ' image')
        image = plt.imread(os.path.join(input_dir, x))
        #print(image.shape)
        f_image = image.flatten()
        #print(f_image.shape)
        #print(f_image)
        data = np.vstack((data, f_image))
        
    #print(data)
    #print(data.shape)
    correct_data = data[1:]
    #print(correct_data)
    #print(correct_data.shape)
    data_t = np.transpose(correct_data)
    #print(data_t)
    #print(data_t.shape)
    data_t = data_t.astype(float)
    #print(data_t)
    #print(data_t.shape)
      
    return data_t
######

'''
 @module:         compress_images(DATA, k)
 @description:    Module to compress images
 @parameters:     DATA = load numpy array of image values
                  k = requested number of PC
 @output:         Output directory 'Output'
'''
def compress_images(DATA,k):
    #  Collect size of DATA matrix
    DATA_shape = DATA.shape
    #print(DATA_shape)
    #  Collect number of input images
    no_Images = DATA_shape[1]
    #print(no_Images)
    
    Z = pca.compute_Z(DATA)
    COV = pca.compute_covariance_matrix(Z)
    #  L = eigenvalues
    #  PCS = eigenvectors
    L, PCS = pca.find_pcs(COV)
    Z_star = pca.project_data(Z, PCS, L, k, 0)
    
    PCS_k = PCS[:k]
    #PCS_transposed = np.transpose(PCS_k)
    #print(Z_star.shape)
    #print(PCS_k.shape)
    X_compressed = np.matmul(Z_star, PCS_k)
    #print(X_compressed)
    #print(X_compressed.shape)
    
    image_Col = np.hsplit(X_compressed, no_Images)
    #print(image_Col)
    #print(image_Col[0].shape)
    #X_compressed is ready to be split back into images and saved
    
    #  Make directory
    path = '/home/student/eclipse_py_workspace/facque_matt_project4/Output'
    isExists = os.path.exists(path)
    if not isExists:
        #  If the directory does NOT exist
        os.mkdir(path)
        #print('Directory Output has been created')
        
    #  Begin rescaling
    min = 0
    max = 0
    counter = 0
    for x in range(no_Images):
        min = np.amin(image_Col[x])
        max = np.amax(image_Col[x])
        
        scaled_Image = ((image_Col[x] - min)/(max - min)) * 255
        true_Image = np.reshape(scaled_Image, dimensions)
        #print(true_Image)
        #print(true_Image.shape)
        filename = 'image'+str(counter)
        complete_Name = os.path.join(path,filename)
        plt.imsave(complete_Name,true_Image,cmap='gray',format='jpg')
        counter += 1
        
    return 0
    
    
    
    