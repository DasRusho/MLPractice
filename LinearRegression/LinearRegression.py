import numpy as np

class LinearRegression():
    def __init__(self, n_epochs = 50, learning_rate = 0.005):
        self. n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features).reshape(n_features,1)
        self.bias = 0

        for i in range(self.n_epochs):
            pred_y = np.dot(X, self.weights) + self.bias
            loss = np.sum((y - pred_y)**2)**0.5

            # check size : X => (n_samples, n_features), y=>(n_samples,1)
            dw = (1/n_samples)*np.dot(X.T,(pred_y-y))
            db = (1/n_samples)*np.sum(pred_y-y)

            self.weights = self.weights - self.learning_rate*dw
            self.bias = self.bias - self.learning_rate*db

            print('Epochs ',i, ' => Loss : ', f'{loss : .3f}')
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias