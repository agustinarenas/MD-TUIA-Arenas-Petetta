import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Sample data generation (replace this with your dataset)
np.random.seed(0)
mean = [0, 0, 0]
cov = [[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]]
data = np.random.multivariate_normal(mean, cov, 100)

# PCA
pca = PCA(n_components=3)
pca.fit(data)
components = pca.components_

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Scatter plot of the original data
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c="b", marker="o", label="Data Points")

# Plot the principal components as vectors
origin = [0, 0, 0]
for i, component in enumerate(components):
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        component[0],
        component[1],
        component[2],
        color="r",
        label=f"Component {i+1}",
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("PCA Class Components in 3D")
ax.legend()

plt.show()
