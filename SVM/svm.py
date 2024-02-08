import numpy as np

def get_l2_norm(x):
    return np.sum(x1**2 for x1 in x)**0.5

class SoftMarginSVMClassifier():
    def __init__(self, n_epochs = 50, param_lambda = 0.01, lr = 0.001):
        self.n_epochs = n_epochs
        self.param_lambda = param_lambda
        self.lr = lr
        self.W = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_svm = np.where(y <= 0, -1, 1)

        self.W = np.zeros(n_features)
        self.b = 0

        for i in range(self.n_epochs):
            margin_loss = get_l2_norm(self.W)**2
            hinge_loss = 0
            delta_w = 0
            delta_b = 0
            for idx, x_i in enumerate(X):
                svm_goal = y_svm[idx] * (np.dot(x_i, self.W) - self.b)
                if svm_goal > 1 : # 0 hinge loss
                    delta_w+= 2*self.param_lambda*self.W
                else:
                    hinge_loss += max(0,1-svm_goal)
                    delta_w+= 2*self.param_lambda*self.W - y_svm[idx]*x_i
                    delta_b+= y_svm[idx]
            svm_loss = margin_loss + (1/n_samples)*hinge_loss
            if i%20 == 0 or i == self.n_epochs - 1: # print loss every once in a while
                print(f'Epoch {i} => Training Loss : {svm_loss}')
            self.W -= self.lr*delta_w
            self.b -= self.lr*delta_b

    def predict(self,X):
        return np.where(np.sign(np.dot(X,self.W) - self.b) < 0, 0, 1)
