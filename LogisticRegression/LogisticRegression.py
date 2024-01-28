import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():
    def __init__(self, learning_rate = 0.0001, n_epochs = 50):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # initialize params
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_epochs):
            # forward propagation
            logits = np.dot(X,self.weights) + self.bias
            pred_y = sigmoid(logits)
            loss = np.sum(-(y*np.log(pred_y) + (1-y)*np.log(1-pred_y)))
            print(f'Epoch {i} => Loss : {loss:.2f}')
            # backpropagation
            dw = (1/n_samples)*np.dot(X.T,(pred_y-y))
            db = (1/n_samples)*np.sum(pred_y-y)
            # weight update
            self.weights -= self.learning_rate*dw
            self.bias -= self.learning_rate*db
    
    def predict(self, X):
        logits = np.dot(X,self.weights) + self.bias
        pred_p = sigmoid(logits)
        pred_y = [1.0 if p > 0.5 else 0.0 for p in pred_p]
        return pred_y

