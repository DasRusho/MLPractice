import numpy as np

def euclideanDistance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def mostFrequentElement(arr):
    count = {}
    for ele in arr:
        count[ele] = count.get(ele,0)+1
    return sorted(count.items(), key = lambda x : x[1], reverse = True)[0][0]

class KNN():
    def __init__(self, k=3):
        self.k=k

    def fit(self,X,y): # Lazy Training
        self.train_X = X
        self.train_y = y

    def predict(self,X):
        predictions = [self.predict_single_data_point(x) for x in X]
        return predictions

    def predict_single_data_point(self,x):
        distances = [euclideanDistance(x_data,x) for x_data in self.train_X]
        knn_idx = np.argsort(distances)[:self.k]
        knn_labels = [self.train_y[idx] for idx in knn_idx]
        return mostFrequentElement(knn_labels)