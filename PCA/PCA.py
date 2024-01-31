import numpy as np

class PCA():
    def __init__(self, n_principal_components = 3):
        self.n_principal_components = n_principal_components
        self.principal_components = None
        self.mean = None
    
    def fit(self,X):
        # mean centre data
        self.mean = np.mean(X,axis=0)
        X = X - self.mean

        # compute covariance matrix , requires samples as columns
        # np.cov applies mean normalization before computing the covariance matrix
        # still we have mean normalized the data in the prev step to maintain parity b/w train and inference
        # np.cov(X.T)  is equivalent to np.dot(X.T,X)/X.shape[1],  assuming X is mean normalized
        cov_mat = np.cov(X.T) # np.dot(X.T,X)/X.shape[1]

        # compute the eigen vectors and eigen values of the covariance matrix
        cov_eigen_vectors, cov_eigen_values = np.linalg.eig(cov_mat)
        cov_eigen_vectors = cov_eigen_vectors.T # make the eigen vectors row vectors

        # Get to #n_principal_components eigen values and the corresponding eigen vectors
        ordered_indices = np.argsort(cov_eigen_values)[::-1]
        self.principal_components = cov_eigen_vectors[ordered_indices[:self.n_principal_components]]
        print('Number of principal components : ',self.n_principal_components)
        print(f'Explained Variance = {np.sum(cov_eigen_values)/np.sum(cov_eigen_values[ordered_indices[:self.n_principal_components]]) : .3f}')

    def project(self,X):
        X = X - self.mean
        return np.dot(X, self.principal_components.T)
