Matt Facque
CS 422
Project4

For this project we were directed to create a PCA algorithm and then a practical application.
The first portion of the project is the file pca.py.  This is the pca algorithm where we were 
directed to create each step of the algorithm culminating in a the return matrix Z_star.  
In the second portion of the project, we were directed to create an image compression algorithm
using the pca algorithm we developed in the first portion of the project.

pca.py
compute_Z:
Module that accepts in an input matrix and outputs that same matrix after having been centered
and scaled.  This is the first part of the PCA algorithm and is simply a matter of pre-processing.
Centering the data is literally to center the data on the origin and is carried out by finding the
mean of each feature set (which for the input matrix is a column) and subtracting the mean from each
item in that column.  Scaling the data is to find the standard deviation of each feature set and 
then to divide each item of the feature set by the standard deviation.  For our purposes, our
algorithm only centers the data but it does not scale it.

compute_covariance_matrix:
Module accepts the centered and scaled matrix and computes the covariance matrix.  The covariance 
matrix is integral to determining the eigenvectors and eigenvalues.  The output of the module is the
covariance matrix which is a square matrix, with shape dependent on the original input matrix.

find_pcs:
Module accepts in the covariance matrix and using in built functions of the numpy library, outputs 
the eigenvectors and eigenvalues of the matrix.  The eigenvectors are in a numpy array together as well
as the eigenvalues.

project_data:
Module that accepts the matrix of eigenvectors, the numpy array of eigenvalues, a value for variance,
a value for the specified number of principal components, and the centered and scaled input matrix.
There are two methods to this module.  The first, if k is specified, means that the user would like
to project the data onto k principal components.  This is carried out using a built in function for 
the numpy library, np.matmul.  The second part of the module is if k is not specified and the user
would like to use as many principal components as needed to cover an amount of variance of the input data.
This is carried out by comparing the required variance to the 'covered' variance of each eigenvalue, 
which is the amount of data that each eigenvalue is accounting for, and simply taking the number of
eigenvectors that suit the specified variance.  The output of the module is Z_star, which is the projected
data.  Since we are using PCA, the projected data can be in fewer dimensions and constitute the most
important features of the data set (since we are using the principal components from largest to smallest).


compress.py
load_data:
Load data is a helper function to load a set of images into a numpy array where the images can then
be compressed using the PCA algorithm.  This method for loading the data is a built in function from
the matplotlib library, imread.  imread outputs an image as a numpy array, which is then flattened
(again, using a built in numpy function), transposed into the correct shape (using a built in numpy
function), cast to floating point values, and returned as the loaded data matrix of all the images
with which to undergo the PCA algorithm.

compress_images:
Module takes in the loaded matrix of image data and the specified number of principal components (k)
that is required and outputs the compressed images in the Output directory.  The method for this 
module is simply to pass the loaded matrix though the functions of pca.py.  That will produce a matrix
X_compressed which is the data of all the images (Z_star) projected onto the k principal components.  The compression
is then completed, the resulting matrix is broken up into the constituent image matrices,
reshaped back into 2-D images and saved into the Output directory.