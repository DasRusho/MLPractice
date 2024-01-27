import numpy as np

class LinearRegressionDataSet():
    def __init__(self, n_features=1, n_samples=100, weights = [5], bias = 2):
        self.n_features = n_features
        self.n_samples = n_samples
        self.weights = weights
        self.bias = bias
        assert(len(weights) == n_features), "Number of weights must be equal to number of features"

    
    def createDataSet(self):
        X = 100*np.random.random_sample((self.n_samples,self.n_features))
        W = np.array(self.weights).reshape(1,self.n_features)
        b = self.bias
        noise = np.random.randn(self.n_samples).reshape(self.n_samples,1)
        y = np.dot(X,W.T) + b + noise
        return X, y