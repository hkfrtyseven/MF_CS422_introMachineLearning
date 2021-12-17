'''
@author:         Matt Facque
@module:         utilities.py
@description:    Python model with helper functions for IMDB database
@import:         aclimdb --> database
'''

import numpy as np
import os 
import random
import string

'''
@Module:        generate_vocab(dir, min_count, max_files)
@Description:   Function to generate the list or numpy of the vocab from the test/train directories
@Parameters:    dir = directory of data
                min_count = minimum number of instances of a word for the word to be added to the vocab
                max_files = number of files to be analyzed for vocab in the target directory
@Output:        list or numpy array of vocabulary
'''
def generate_vocab(dir, min_count, max_files):
    #  Dictionary to preprocess vocabulary
    preprocess_Dict = dict()
    
    #  Append /pos and /neg to directory
    directory = dir
    pos_directory = directory + "/pos/"
    neg_directory = directory + "/neg/"
    
    #print(dir)
    #print(pos_directory)
    #print(neg_directory)
    
    #print(os.getcwd())
    
    #  Number of files to read
    if (max_files < 0):
        #  Read all input files to create vocab
        #  Process 'pos' reviews
        for file in sorted(os.listdir(pos_directory)):
            text = open(pos_directory + file, 'r')
            for line in text:
                #  Preprocessing
                line = line.strip()
                line = line.lower()
                line = line.translate(line.maketrans("", "", string.punctuation))
                words = line.split(" ")
                for word in words:
                    if word in preprocess_Dict:
                        preprocess_Dict[word] = preprocess_Dict[word] + 1
                    else:
                        preprocess_Dict[word] = 1
            text.close()
            
        #  Process 'neg' reviews
        for file in sorted(os.listdir(neg_directory)):
            text = open(neg_directory + file, 'r')
            for line in text:
                #  Preprocessing
                line = line.strip()
                line = line.lower()
                line = line.translate(line.maketrans("", "", string.punctuation))
                words = line.split(" ")
                for word in words:
                    if word in preprocess_Dict:
                        preprocess_Dict[word] = preprocess_Dict[word] + 1
                    else:
                        preprocess_Dict[word] = 1
            text.close()
        
    else:
        half = max_files / 2
        pos_files = int(half)
        #print(pos_files)
        neg_files = int(max_files - half)
        #print(neg_files)
        
        #  Process 'pos' reviews
        counter = 0
        for file in sorted(os.listdir(pos_directory)):
            #print(file)
            text = open(pos_directory + file, 'r')
            for line in text:
                #  Preprocessing
                line = line.strip()
                line = line.lower()
                line = line.translate(line.maketrans("", "", string.punctuation))
                words = line.split(" ")
                for word in words:
                    if word in preprocess_Dict:
                        preprocess_Dict[word] = preprocess_Dict[word] + 1
                    else:
                        preprocess_Dict[word] = 1
            text.close()
            counter += 1
            if counter == pos_files:
                break
        
        #  Process 'neg' reviews
        counter = 0
        for file in sorted(os.listdir(neg_directory)):
            text = open(neg_directory + file, 'r')
            for line in text:
                #  Preprocessing
                line = line.strip()
                line = line.lower()
                line = line.translate(line.maketrans("", "", string.punctuation))
                words = line.split(" ")
                for word in words:
                    if word in preprocess_Dict:
                        preprocess_Dict[word] = preprocess_Dict[word] + 1
                    else:
                        preprocess_Dict[word] = 1
            text.close()
            counter += 1
            if counter == neg_files:
                break
    
    #for key in list(preprocess_Dict.keys()):
    #    print(key, ":", preprocess_Dict[key])
    #  Create output list
    output_List = []
    
    for key, value in preprocess_Dict.items():
        if value >= min_count:
            output_List.append(key)
    output_List.remove('')
    #print(len(output_List))
    #print(output_List)
                
    return output_List
######

'''
@Module:        create_word_vector(fname, vocab)
@Description:   Function to create a feature vector using a file and the vocab
@Parameters:    fname = the filename of the file to be analyzed
                vocab = the vocabulary to use in the analysis
@Output:        array (feature vector)
'''
def create_word_vector(fname, vocab):
    text = open(fname, 'r')
    
    #  Create feature vector
    feat_vec = []
    for line in text:
        feat_vec.append([1 if word in line.lower().translate(line.maketrans("", "", string.punctuation)).split(" ") else 0 for word in vocab])
    
    text.close()
    #print(feat_vec)
        
    return np.asanyarray(feat_vec)
######

'''
@Module:        load_data(dir, vocab, max_files)
@Description:   Creates a list/array of feature vectors with associated labels
@Parameters:    dir = directory of reviews to make feature vectors
                vocab = list of words that make up feature vector
                max_files = the maximum number of files to use to create the feature vector
                NOTE:  The number of columns will be LENGTH OF THE VOCAB 
                        and the number of feature vectors is the NUMBER OF MAX_FILES
@Output:        X = list/array of feature vectors
                Y = list/array of associated labels
'''
def load_data(dir, vocab, max_files):
    #  Append /pos and /neg to directory
    directory = dir
    pos_directory = directory + "/pos/"
    neg_directory = directory + "/neg/"
    
    #  Initialize empty np arrays
    vector_Feat = np.empty(len(vocab))
    #print(vector_Feat)
    vector_Label = np.empty(0)
    #print(vector_Label)
    
    if (max_files < 0):
        #  Process 'pos' files
        for file in sorted(os.listdir(pos_directory)):
            #print(file)
            text = pos_directory + file
            feature_vector = create_word_vector(text, vocab)
            np.vstack((vector_Feat, feature_vector))
            np.append(vector_Label, [1])

        #  Process 'neg' files
        for file in sorted(os.listdir(neg_directory)):
            text = neg_directory + file
            feature_vector = create_word_vector(text, vocab)
            np.vstack((vector_Feat, feature_vector))
            np.append(vector_Label, [0])
            
        vector_Feat = np.delete(vector_Feat, 0)
        return vector_Feat, vector_Label
    
    else:
        half = max_files / 2
        pos_files = int(half)
        #print(pos_files)
        neg_files = int(max_files - half)
        #print(neg_files)
        
        #  Process 'pos' files
        counter = 0
        for file in sorted(os.listdir(pos_directory)):
            #print(file)
            text = pos_directory + file
            feature_vector = create_word_vector(text, vocab)
            #print(feature_vector)
            vector_Feat = np.vstack((vector_Feat, feature_vector))
            vector_Label = np.append(vector_Label, [1])
            counter += 1
            if counter == pos_files:
                break
        #print(counter)
        
        #  Process 'neg' files
        counter = 0
        for file in sorted(os.listdir(neg_directory)):
            text = neg_directory + file
            feature_vector = create_word_vector(text, vocab)
            vector_Feat = np.vstack((vector_Feat, feature_vector))
            vector_Label = np.append(vector_Label, [0])
            counter += 1
            if counter == neg_files:
                break
        #print(counter)
        
        vector_Feat = np.delete(vector_Feat, 0, 0)    
        #print(vector_Feat)
        #print(vector_Feat.shape)
        #print(vector_Label)
        #print(vector_Label.shape)
        return vector_Feat, vector_Label
    
######        