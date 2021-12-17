Matt Facque
Project 5
CS 422
scikit models

Utilities.py
Utilities.py contains the functions to process and load the feature/label data.
The helper functions are generate_vocab, create_word_vector, and load_data.
Utilities.py works in conjunction with the IMDB acl dataset of positive and negative
reviews.  The methodology for creating the feature set is the bag of words method
from the positive and negative reviews.  The vocabulary created from the reviews is then
used to create the feature vector, a vector of 1's and 0's of whether or not the review
contains the word (1) or not (0).  The label set is taken from the positive and negative 
reviews.  Positive reviews are given the label 1 and negative reviews are given the label 0.

generate_vocab(dir, min_count, max_files)
This function goes through a specified number of positive and negative reviews creating a 
vocabulary of words (bag of words method).  The word chosen for the vocabulary is taken when 
the number of instances of the word is greater than a predefined number.  For example, we parse 
through a number of positive and negative reviews and add every word that is encountered 2 or more
times to the vocabulary.  Also, the number of positive and negative reviews is equal and determined
by the programmer, this is the variable max_files.  The reviews receive preprocessing in the form of
string manipulations like removal of white space, removal of punctuations, and forcing all words
into lowercase.  This makes our vocabulary a list of words in lowercase without any misleading 
characters.  The function returns a list of all chosen vocab words.

create_word_vector(fname, vocab)
This function takes a file, a review in our case, and a vocabulary and creates a feature vector.
The filename is passed to the function, the file is processed similarly to in generate_vocab()
and the file is parsed searching for words in the vocabulary.  The feature vector is made up 
of 1's if the word is present in the review or 0's if the word is not present in the review.
The feature vector in the form of a numpy array is returned.

load_data(dir, vocab, max_files)
This function processes equal number of positive and negative reviews using create_word_vector()
creating the total feature set.  During this process, the label set is also created.  After completion
of the processing, the function returns the feature set and label set.  The feature set and the label set
are built from the vocabulary, positive, and negative reviews and are ready for use in ML models.

ml.py
ml.py contains the model creation, model testing, and f1 score from the models.  Importantly, ml.py uses 
the scikit-learn library for creation of the machine learning models, the testing of the models (generating
predictions), and for determining the f1 score.  The f1 score is the harmonic mean of precision recall where 
the value is the best at 1 and the worst at 0.  The creation of the models is done completely with the scikit-learn
library in conjunction with the training feature and label set (created in utilities.py) and the testing feature
and label set (similary created in utilities.py).

model_test(X, model)
This function uses the scikit-learn predict() method to make a prediction on the passed in model.  X is the 
testing feature set and model is the machine learning model to produce a prediction on.  The function returns the 
prediction from the model.

compute_f1(Y, Y_hat)
This function produces the f1 score using the label set which is the true label set from the analyzed data
and Y_hat which is the predicted label set from the model_test function.  The value returned from the 
function is a value between 0 and 1 representing the harmonic mean between precision recall.  This function is used
in conjunction with model_test and is therefore the f1 score of whichever model is passed to model_test.

dt_train(X, Y)
This function produces a trained decision tree model.  X is the feature set and Y is the label set.  Since decision trees 
are supervised machine learning models, the function requires the feature set and label set together.

pca_train(X, K)
This function produces a feature set with reduced dimensionality from the input data set X.  X initially is the full
size feature set and is returned as the reduced feature set.  K is the number of requested components in the output
feature set.

pca_transform(X, pca)
This function performs PCA on a dataset (X) using the reduced dataset PCA (pca) which was completed in the pca_train
function.  Generally, the dataset that is passed to pca_train is the training dataset which produces a transformation.
Then, in pca_transform, the training dataset and testing dataset is transformed using the previously gained transformation.

kmeans_train(X)
This function creates a kmeans clustering model on the feature set X and returns the model.

knn_train(X, Y, K)
This function plots the data using the feature set X with the input label set Y and will allow for predictions using
K nearest neighbors.  The function returns the fitted K nearest neighbor model.

preceptron_train(X, Y)
This function creates a perceptron using the input feature set X and label set Y and returns the trained perceptron.

nn_train(X, Y, hls)
This function creates a neural network with a number of hidden layers equalling hls.  hls is a tuple where the ith value
is the number of perceptrons in the ith layer.  The neural network is trained using the input feature set X and label set
Y.  The function returns the trained neural network model.

svm_train(X, Y, k)
This function creates a support vector machine using the input feature set X and label set Y.  k is the input kernel.  
The function returns the fitted svm. 
















