DT_train_binary:
 -> Decision tree is implemented in a binary tree using a Node class (DT_Node in decision_trees.py).
 
 -> DT_train_binary uses parameters X (feature set), Y (label set), max_depth (operator requested depth of decision tree).
 
 -> DT_train_binary initializes the decision tree and calls a function build_Tree(list_X, list_Y, depth, set_Ent, node_Label).
 
 -> build_Tree is a recursive function that calculates the information gain (IG) using the feature set (list_X) and the label set (list_Y),
 		uses the base case depth == 0 or if leaf IG == 0/1.  Build_Tree then recursively populates the left and right child nodes.
 
 -> build_Tree
 -> First:  Test base case 1 (depth == 0)
 -> Next: Test base case 2 (IG of leaf == 0/1)
 -> Next: Determine the feature with which to split the feature set (value in the node to determine the decision)
 -> Next: helper function split_Data (return the feature value to split the feature set in the decision tree)
 				accepts the feature set, label set, and feature to split and
 				returns the feature set and label set that will travel down the left side of the decision tree (no side) and
 				the feature set and label set that will travel down the right side of the decision tree (yes side)
 -> Next: Initialize the current node with the feature value to split, the node label (+1/-1 depending on characteristic of decision tree)
 -> Next: Recursively call build_Tree for the current_node.left and the current_node.right using the feature sets and label sets previously
 				determined to travel down the left and right sides of the tree respectively
 
 -> Decision tree will be implemented as a binary tree at the end of the recursion when a base case is triggered.
 
 -> helper function: determine_IG (function to return the IG of the leaf node, assist in base case 2)
 				accepts the feature set and label sets passed into the node as well as the total set entropy (H()),
 				in a for loop, process feature set and label set to determine leaf entropy
 				calculate total IG for the leaf and if the IG is 0 or 1, return 0 (which indicates a base case)
 				
 -> helper function: split_Feature (function to return the feature that when split will have the highest IG)
 				accepts the current node feature set and label set
 				in a for loop, process feature set and label set to determine which feature is best split
 				returns integer value of feature
 -> helper function: IG_calc (splits feature set and label set into yes/no features with corresponding labels)
 				function called in split_Feature, used to assist in calculating IG
 				accepts the feature set, label set, and the total set entropy (H())
 				using zip() function, calculates the yes features and no features of the chosen split_Feature
 				returns total IG = H() - (sum of probability of node() + entropy of node())
 -> helper function: calculate_LeafIG (math function to return probability of node + entropy of node)
 				function called in IG_calc, returns node value of probability of node + entropy of node
 				accepts the number of items in node, number of 0 labels (no labels), number of 1 labels (yes labels),
 				and the total number of items in the node
 				performs mathematical calculations to get the probability of landing in a sub node (yes/no sub node) and 
 				the log function of the sub node
 				returns the sum of the probability of the node + the entropy of the node
 	NOTE:
 	The decision tree algorithm proved to be the most difficult to implement.  The first portion, calculating information gain
 	seemed to work fairly easily but implementing the decision tree was difficult.  The decision tree is implemented recursively
 	which proved to be fairly efficient and the simplest method.  The decision tree produced by the algorithm is correct in form but 
 	the accuracy from the tree may be suspect.  I would need to work on the algorithm more to include the special cases as the algorithm 
 	now is fairly rudimentary.
 				
DT_test_binary:
 -> accepts a test set of features and a test set of labels and the trained decision tree
 -> For loop to traverse the trained decision tree the number of times that corresponds with the number of samples
 -> For loop returns an accuracy count that is used to return the exact accuracy for the test set of data
 -> helper function:  tree_Traverse
 				Travels through the decision tree, testing against the feature value of the node to the feature set of the data
 				traversing either the left or right node depending on the value therein (yes/no)
 				The base case of the recursion is when the function has traversed to a leaf and returns the value of that leaf node
 				At the end of the recursion, the return value is passed back up the chain to be stored in the accuracy variable
 				
DT_make_prediction:
 -> accepts a feature data set and the trained decision tree
 -> recursively travels through the tree until the base case
 -> the leaf node of the decision represents the base case of the recursion, when the base case of the function is met, the label value in the leaf
 	node is passed back up the tree to make a prediction using the input feature set
 	
KNN_test:
 -> accepts a training feature set, training label set, testing feature set, testing label set, and K (number of nearest neighbors)
 -> function produes a [training feature set] x [testing feature set] matrix of distances (distances from each point to each other point)
 -> a nested for loop is used to traversed the distance array (the matrix of distances)
 -> when the lowest distance is found (the nearest neighbor), the label for that point is added to a temporary list of labels
 -> a value for accuracy is then gained from the mix of labels produced from the labels of the nearest neighbors
 -> accuracy is divided by the total number of samples in the feature set and that represents the total accuracy for the set
	NOTE:
	This is a basic version of K-Nearest Neighbors.  I simply ran out of time to produce better results.  It seems the algorithm works for K=1 
	which, while being the basic case, is also underfitting and the model is incapable of accurately checking complex data sets.
 
 
 
 
 
 
 
 
 
 ->  