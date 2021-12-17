'''
@author: Matt Facque
'''

import numpy as np
import math

#  Class for Binary Tree
class DT_Node:
    def __init__(self, Ldata, Fdata, feature_Set, label_Set):
        self.left = None
        self.right = None
        self.label_Val = Ldata
        self.feature_Val = Fdata
        self.feature_Set = feature_Set
        self.label_Set = label_Set
        
    def print_Tree(self):
        if self.left:
            self.left.print_Tree()
        print(self.feature_Val)
        if self.right:
            self.right.print_Tree()
            
    
######  Class for Binary Tree
        
#  Import test data sets into program
def import_file(filename):
    f = open(filename, 'r')
    counter = 0
    label_string = ''
    for line in f:
        line = line.strip().split(';')
        if counter == 0:
            feature_string = line
        else:
            label_string = line  
        counter+=1
        
    f.close()

#print(feature_string)
#print(label_string)
    
    Feature = []
    Label = []
    
    for value in feature_string:
        temp = [float(val) for val in value.split(',')]
        Feature.append(temp)
    
    if len(label_string)>0:
        for value in label_string:
            temp = int(value)
            Label.append(temp)
            
    X = np.array(Feature)
    Y = np.array(Label)
    
    return X, Y
###### import_file

#  Determine entropy on list of values
#  Argument list(set) is set of features or values to obtain entropy
#  No = entropy of label = 0/-1/No
#  Yes = entrypy of label = 1/1/Yes
def calc_entropy(val_List):
    #  Initialize variables
    no_log = 0
    yes_log = 0
    Yes = 0
    No = 0
    counter = 0
    for x in val_List:
        counter+=1
        if x > 0:
            Yes+=1
        else:
            No+=1
    #  If val_List is empty, for loop didn't go, counter == 0
    if (counter == 0):
        return 0
    
    #  calculate Yes/No probability
    no_prob = No / counter
    yes_prob = Yes / counter
    #  calculate log2(probability(Yes/No)
    #  If either probability = 0, log = 0
    if (no_prob == 0):
        no_log = 0
    elif (yes_prob == 0):
        yes_log = 0
    else:
        no_log = math.log2(no_prob)
        yes_log = math.log2(yes_prob)
    #  Yes/No entropy
    no_entropy = (-1 * no_prob) * no_log
    yes_entropy = (-1 * yes_prob) * yes_log
    #  Calculate entropy of total set
    H_of_set = no_entropy + yes_entropy
    
    return H_of_set
###### entropy of set of values

#  Calculate entropy for sub node (leaf)
#  i_Innode = items in node
#  node_Labels = int value for labels (node_Labels = number of 1's predicted at node(leaf))
#  total_Items = total items in set
def calculate_LeafIG(i_InLeaf, node_Label_no, node_Label_yes, total_Items):
    #  Total number of items in leaf
    leaf_Prob = i_InLeaf / total_Items
    
    #print(i_InLeaf)
    #print(leaf_Prob)
    
    #if i_InLeaf == 0:
    #   no_prob = 0
    #   yes_prob = 0
    
    no_Labels = node_Label_no
    #print(no_Labels)
    yes_Labels = node_Label_yes
    #print(yes_Labels)
    #  calculate Yes/No probability
    if i_InLeaf == 0:
        no_prob = 0
        yes_prob = 0
    else:
        no_prob = no_Labels / i_InLeaf
        yes_prob = yes_Labels / i_InLeaf
        
    #print(no_prob)
    #print(yes_prob)
    #  calculate log2(probability(Yes/No))
    if no_prob == 0:
        no_log = 0
    else:
        no_log = math.log2(no_prob)
    if yes_prob == 0:
        yes_log = 0
    else:
        yes_log = math.log2(yes_prob)
    #  Yes/No subLeaf IG
    no_entropy = (-1 * no_prob) * no_log
    yes_entropy = (-1 * yes_prob) * yes_log
    #  Check entropy calculation
    leaf_entropy = no_entropy + yes_entropy
    #print(leaf_entropy)
    #  Calculate subLeaf IG
    IG_of_Leaf = leaf_Prob * (leaf_entropy)
    
    #print(IG_of_Leaf)
    return IG_of_Leaf
###### calculate_Subentropy

#  Split feature determines what feature to split data set on
#  using information gain
def split_Feature(feat, label):
    if (len(feat)):
        data_Set_count = 0
        for x in feat[0]:
            data_Set_count+=1
    else:
        data_Set_count = 0
        
    #  Calculate entropy of entire set
    H = calc_entropy(label)
    #  List of extracted feature values
    feature_Set = []
    #  List of information gain, position corresponds to feature
    information_Gain = []
    #  feature_Val will be feature with best IG
    feature_Val = 0
    
    for x in range(data_Set_count):
        feature_Set = extraction(feat, feature_Val)
        #print(feature_Set)
        gain_Value = IG_calc(feature_Set, label, H)
        information_Gain.append(gain_Value)
        feature_Set.clear()
        feature_Val+=1
    
    #  Determine max IG and corresponding feature
    max_Val = 0
    split_Val = 0
    feature_Count = 0
    for x in information_Gain:
        #print(x)
        if x > max_Val:
            max_Val = x
            split_Val = feature_Count
        feature_Count+=1    
        
    return split_Val
######  split_Feature

#  determine_IG determines what the IG of a leaf is
#  using similar functionality as split_Feature
def determine_IG(feat, label, entropy):
    if (len(feat)):
        data_Set_count = 0
        for x in feat[0]:
            data_Set_count+=1
    else:
        data_Set_count = 0

    #  Calculate entropy of entire set
    H = entropy
    #  List of extracted feature values
    feature_Set = []
    #  List of information gain, position corresponds to feature
    information_Gain = []
    #  feature_Val will be feature with best IG
    feature_Val = 0
    
    for x in range(data_Set_count):
        feature_Set = extraction(feat, feature_Val)
        #print(feature_Set)
        gain_Value = IG_calc(feature_Set, label, H)
        information_Gain.append(gain_Value)
        feature_Set.clear()
        feature_Val+=1
    
    test = False
    
    for y in information_Gain:
        if (y == 0):
            test = True
        else:
            test = False
    for y in information_Gain:
        if (y == 1):
            test = True
        else:
            test = False
    
    if (test == True):
        test = 0
    else:
        test = 2
            
    return test
######  determine_IG

#  Extract ith element of sublist
def extraction(subList, pos):
    return [item[pos] for item in subList]
###### extraction

#  IG_calc is used with a feature set, the label set, H to calculate IG
#  target_Set = set of values for feature[#]
#  lab_Set = set of labels
#  total_e = total set entropy
def IG_calc(target_Set, lab_Set, total_e):
    #print(target_Set)
    #print(lab_Set)
    #  Check if target_Set has values, if no = return 0
    if (len(target_Set) == 0):
        return 0
    
    #  Total items to traverse
    total_Items = len(lab_Set)
    #  Combine target_Set list with lab_Set list
    paired_List = list(zip(target_Set, lab_Set))
    #print(paired_List)
    
    left_BranchVal = 0
    right_BranchVal = 0
    left_BranchLabel_no = 0
    left_BranchLabel_yes = 0
    right_BranchLabel_no = 0
    right_BranchLabel_yes = 0
    
    #print(paired_List[0])
    for i in paired_List:
        if i[0] == 0 and i[1]== 0:
            left_BranchVal+=1
            left_BranchLabel_no+=1
        elif i[0] == 1 and i[1] == 0:
            right_BranchVal+=1
            right_BranchLabel_no+=1
        elif i[0] == 0 and i[1] == 1:
            left_BranchVal+=1
            left_BranchLabel_yes+=1
        elif i[0] == 1 and i[1] == 1:
            right_BranchVal+=1
            right_BranchLabel_yes+=1
    
    #print("Left node labels = " + str(left_BranchLabel))
    #print("Right node labels = " + str(right_BranchLabel))
    
#  BranchVal = number of items in leaf
#  BranchLabel_no = number of labels = 0 in leaf
#  BranchLabel_yes = number of labels = 1 in leaf
#  total_Items = number of total items in feature set

    left_NodeEnt = calculate_LeafIG(left_BranchVal, left_BranchLabel_no, left_BranchLabel_yes, total_Items)
    right_NodeEnt = calculate_LeafIG(right_BranchVal, right_BranchLabel_no, right_BranchLabel_yes, total_Items)
    
    total_InformationGain = total_e - (left_NodeEnt + right_NodeEnt)
         
    return total_InformationGain
###### IG_calc

#  split_Data
#  list_Feat = feature set
#  list_Lab = label set
#  feature = feature to split data on
#  Combines feature set and label set into ([xxxx], x) and splits data
def split_Data(list_Feat, list_Lab, feature):
    lfeature = []
    llabel = [] 
    rfeature = []
    rlabel = []
    
    full_List = list(zip(list_Feat, list_Lab))
    
    for i in full_List:
        if (i[0][feature] == 1):
            rfeature.append(i[0])
            rlabel.append(i[1])
        else:
            lfeature.append(i[0])
            llabel.append(i[1])
    
    return lfeature, llabel, rfeature, rlabel
######  split_Data

#  DT_train_binary: takes in feature, label list and max_depth to build tree
def DT_train_binary(X, Y, max_depth):
    #  Variable to store desired depth
    tree_Depth = 0
    #  Variable to store # of features
    data_Set_count = 0
    for x in X[0]:
        data_Set_count+=1
    #  Variable to hold total set entropy
    H = calc_entropy(Y)
        
    if (max_depth == 0):
        print("Error, Tree Depth = 0\n")
        return 0
    elif (max_depth == -1):
        tree_Depth = data_Set_count
    else:    
        tree_Depth = max_depth
    
    #print("DT begin")
    #  Make feature set and label set into 2-d list and 1-d list
    list_X = X.tolist()
    list_Y = Y.tolist()
    root = build_Tree(list_X,list_Y,tree_Depth,H,0)
    
    return root
######  DT_train_binary

#  build_Tree
def build_Tree(list_X, list_Y, depth, set_Ent, node_Label):
    #print(list_X)
    #print(list_Y)
    
    #  Label for leaf using any possible data in node
    temp_Label = 0
    
    #  1st base case
    #print("base case 1")
    if (depth == 0):
        if (len(list_Y)):
            for i in list_Y:
                if (i == 0):
                    temp_Label -= 1
                else:
                    temp_Label += 1
            if (temp_Label > 0):
                return DT_Node(1, 'x', [], [])
            else:
                return DT_Node(-1, 'x', [], [])
        else:
            if (node_Label == -1):
                return DT_Node(-1, 'x', [], [])
            else:
                return DT_Node(1, 'x', [], [])
    #  2nd base case
    #print("base case 2")
    IG_leaf = determine_IG(list_X, list_Y, set_Ent)
    if (IG_leaf == 0):
        if (len(list_Y)):
            for i in list_Y:
                if (i == 0):
                    temp_Label -= 1
                else:
                    temp_Label += 1
            if (temp_Label > 0):
                return DT_Node(1, 'x', [], [])
            else:
                return DT_Node(-1, 'x', [], [])
        else:
            if (node_Label == -1):
                return DT_Node(-1, 'x', [], [])
            else:
                return DT_Node(1, 'x', [], [])
    
    #print("tree growth")
    feature_To_split = split_Feature(list_X, list_Y)
    #print(feature_To_split)
    
    left_Feature, left_Label, right_Feature, right_Label = split_Data(list_X, list_Y, feature_To_split)
    #print(left_Feature)
    #print(left_Label)
    #print(right_Feature)
    #print(right_Label)
    
    #print("Build next level of tree")
    node = DT_Node(node_Label, feature_To_split, [], [])
    node.left = build_Tree(left_Feature, left_Label, depth-1, set_Ent, -1)
    node.right = build_Tree(right_Feature, right_Label, depth-1, set_Ent, 1)
    return node
######  build_Tree    

#  Dt_test_binary
#  X = test feature set
#  Y = test label set
#  decisionTree = trained decision tree
def DT_test_binary(X, Y, decisionTree):
    #  Variable for keeping track of test accuracy
    accuracy = 0
    #  Variable for accuracy versus tree
    total_Acc = 0
    #  Change X and Y to lists for simplicity
    list_X = X.tolist()
    list_Y = Y.tolist()
    #  traverse decision tree however many labels are in test set
    count_Labels = len(list_Y)
    for i in range(count_Labels):
        accuracy += tree_Traverse(list_X[i],list_Y[i],decisionTree)
    
    total_Acc = accuracy / count_Labels    
    
    return total_Acc
######  Dt_test_binary

#  tree_Traverse
#  X = one sample of feature set
#  Y = corresponding label to sample of feature set
#  dt = trained decision tree
def tree_Traverse(X, Y, dt):
    #  base case, 'x' denotes leaf
    if (dt.feature_Val == 'x'):
        if (dt.label_Val == 1):
            if (Y == 1):
                return 1
            else:
                return 0
        else:
            if (Y == 0):
                return 1
            else:
                return 0
    
    accuracy_count = 0
    
    if (X[dt.feature_Val] == 1):
        accuracy_count = tree_Traverse(X, Y, dt.right)
    else:
        accuracy_count = tree_Traverse(X, Y, dt.left)
        
    return accuracy_count
######  tree_Traverse

#  DT_make_prediction(x, dt)
#  x = prediction data set
#  dt = trained decision
def DT_make_prediction(x, dt):         
    if (dt.feature_Val == 'x'):
        return dt.label_Val
    
    prediction = 0
    
    if (x[dt.feature_Val] == 1):
        prediction = DT_make_prediction(x, dt.right)
    else:
        prediction = DT_make_prediction(x, dt.left)
        
    return prediction
######  DT_make_prediction
    
    
#X_features, Y_labels = import_file("data_2.txt")
#print(X_features)
#print(Y_labels)
#X_test, Y_test = import_file("data_2.txt")
#X_pred, Y_pred = import_file("pred_1.txt")
#feature_to_begin_DT = split_Feature(X_features, Y_labels)
#print(feature_to_begin_DT)
#depth = 3
#dt = DT_Train_Binary(X_features,Y_labels, depth)
#dt.print_Tree()        
#accuracy = DT_test_binary(X_test, Y_test, dt)     
#print(accuracy)
#prediction = DT_make_prediction(X_pred, dt)
#print(prediction)

       
        
        
        
        
        
        
        
        
        
        