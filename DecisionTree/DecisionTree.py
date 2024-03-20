import numpy as np
from collections import Counter

class TreeNode():
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):
        self.feature = feature  # feature to split
        self.threshold = threshold # threshold of feature value to split
        self.left = left
        self.right = right
        self.valueIfLeafNode = value
    def is_leaf_node(self):
        return self.valueIfLeafNode is not None
        


class DecisionTreeClassifier():
    def __init__(self, root = None, n_features = None, max_depth = 100, min_samples_to_split = 8):
        self.root = root
        self.n_features = n_features
        self. max_depth = max_depth
        self.min_samples_to_split = min_samples_to_split

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_features = n_features if self.n_features is None else min(n_features,self.n_features)
        self.root = self._grow_tree(X, y)

    # recursive function to build the tree
    def _grow_tree(self, X, y, depth = 0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        # check for the stopping criteria
        if depth == self.max_depth or n_samples < self.min_samples_to_split or n_labels == 1:
            leaf_value = self._majority_label(y)
            return TreeNode(value = leaf_value)


        # find the best feature to split on and the corresponding threshold
        idx_features_to_select_from = np.random.choice(n_features, self.n_features, replace = False)

        split_feature, split_threshold = self._find_best_split(X, y, idx_features_to_select_from)


        # create left and right children based on the split
        left_idxs, right_idxs = self._split_node(X[:,split_feature], split_threshold)
        left_child = self._grow_tree(X[left_idxs, :],y[left_idxs],depth+1)
        right_child = self._grow_tree(X[right_idxs, :],y[right_idxs],depth+1)

        return TreeNode(split_feature, split_threshold, left_child, right_child)
    
    def _find_best_split(self,X, y, idx_features_to_select_from):
        idx_winner_feature, split_threshold = None, None 
        best_gain = -1

        for feat_idx in idx_features_to_select_from:
            X_feature = X[:,feat_idx]
            X_feature_values = np.unique(X_feature)

            for threshold in X_feature_values:
                # compute information gain
                gain = self._information_gain(X_feature, y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    idx_winner_feature = feat_idx
                    split_threshold = threshold

        return idx_winner_feature, split_threshold
    
    def _information_gain(self,X_feature, y, threshold):
        # compute node entropy
        node_entropy = self._compute_entropy(y)

        # create children based on threshold
        idx_left_child, idx_right_child = self._split_node(X_feature, threshold)
        if len(idx_left_child) == 0 or len(idx_right_child) == 0:
            return 0

        # compute weighted avg of entropies of children nodes
        left_child_n_samples, left_child_entropy = len(idx_left_child), self._compute_entropy(y[idx_left_child])
        right_child_n_samples, right_child_entropy = len(idx_right_child), self._compute_entropy(y[idx_right_child])
        children_entropy_weighted_avg = left_child_n_samples*left_child_entropy/len(y)
        children_entropy_weighted_avg += right_child_n_samples*right_child_entropy/len(y)

        # compute information gain
        information_gain = node_entropy - children_entropy_weighted_avg

        return information_gain
    
    def _split_node(self,X_feature, threshold):
        idx_left_child = np.argwhere(X_feature <= threshold).flatten()
        idx_right_child = np.argwhere(X_feature > threshold).flatten()
        return idx_left_child, idx_right_child

    def _compute_entropy(self,y):
        y_counts = np.array(list(Counter(y).values()))
        total_count = len(y)
        ps = y_counts/total_count
        entropy = -1*np.sum([p* np.log(p) for p in ps if p>0])
        return entropy


    

    def _majority_label(self, y):
        return sorted(Counter(y).items(), key = lambda x : x[1], reverse = True)[0][0]

    

    def predict(self, X):
        return np.array([self._run_tree(x, self.root) for x in X])
    
    def _run_tree(self, x, node):
        if node.is_leaf_node():
            return node.valueIfLeafNode
        if x[node.feature] <= node.threshold:
            return self._run_tree(x, node.left)
        return self._run_tree(x, node.right)



