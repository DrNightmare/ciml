import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets


def get_closest_clusters(examples, centers):
    return np.argmin(np.linalg.norm(examples - centers[:, None], axis=2), axis=0)


# Implementation of Algorithm 4 - K-Means
# returns cluster assignments and cluster centers (useful for plotting)
def kmeans(data, k):
    # randomly initialize center for kth cluster
    cluster_centers = data[np.random.choice(data.shape[0], 3, replace=False)]
    mean_changed = True

    cluster_assignments = None

    while mean_changed:
        # assign example n to closest cluster
        cluster_assignments = get_closest_clusters(data, cluster_centers)
        mean_changed = False
        for i in range(k):
            # points assigned to cluster
            points_in_cluster = np.array([example for index, example in enumerate(data) if cluster_assignments[index] == i])
            if points_in_cluster.size:
                old_cluster_center = cluster_centers[i].copy()

                # reestimate centers of cluster
                cluster_centers[i] = points_in_cluster.mean(axis=0)

                if np.linalg.norm(old_cluster_center - cluster_centers[i]) > 0.0005:
                    mean_changed = True

    return cluster_assignments, cluster_centers


x, y = datasets.load_iris(return_X_y=True)
labels, centers = kmeans(x, 3)

fig = plt.figure()

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

ax.scatter(x[:, 3], x[:, 1], x[:, 2], c=labels.astype(np.float), edgecolor='k')
ax.scatter(centers[:, 3], centers[:, 1], centers[:, 2], c='r', marker='*')

ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')

plt.show()
