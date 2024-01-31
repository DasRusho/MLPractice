import matplotlib.pyplot as plt
from sklearn import datasets
from PCA import PCA

# load iris dataset
data = datasets.load_iris()
X = data.data
y = data.target

# project data onto principal components
n_principal_components = 2
pca = PCA(n_principal_components=n_principal_components)
pca.fit(X)
X_dim_reduced = pca.project(X)

print("Shape of original data :", X.shape)
print("Shape of transformed data:", X_dim_reduced.shape)

x1 = X_dim_reduced[:, 0]
x2 = X_dim_reduced[:, 1]

plt.scatter(
    x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()
