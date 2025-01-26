"""
----- Using Numpy -----
"""
import random as r
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

COLORS = ['green', 'blue', 'black', 'purple']
MAX_X = 500
MAX_Y = 200
CENTROIDS = np.random.randint(3, 4)
POINTS_PER_CENTROID = 300

centroids_cord = np.array([
    [
        r.randint(-MAX_X + MAX_X // 5, MAX_X - MAX_X // 5),
        r.randint(-MAX_Y + MAX_Y // 5, MAX_Y - MAX_Y // 5)
    ]
    for _ in range(CENTROIDS)
])

rand_clst_points = []
for centroid in centroids_cord:
    cluster_points = [
        [
            r.randint(centroid[0] - MAX_X // 5, centroid[0] + MAX_X // 4),
            r.randint(centroid[1] - MAX_Y // 4, centroid[1] + MAX_Y // 4)
        ]
        for _ in range(POINTS_PER_CENTROID)
    ]
    rand_clst_points.extend(cluster_points)

rand_clst_points = np.array(rand_clst_points)

def assign_points(points, centroids):
    distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def calculate_new_centroids(points, labels, num_centroids):
    new_centroids = []
    for i in range(num_centroids):
        cluster_points = points[labels == i]
        if len(cluster_points) > 0:  # Avoid division by zero
            new_centroids.append(np.mean(cluster_points, axis=0))
        else:
            new_centroids.append([0, 0])  # Fallback for empty clusters
    return np.array(new_centroids)

def k_means(points, initial_centroids, iterations):
    centroids = initial_centroids
    for _ in range(iterations):
        labels = assign_points(points, centroids)
        centroids = calculate_new_centroids(points, labels, len(centroids))
    return centroids, labels

def plot_clusters(points, initial_centroids, final_centroids, labels):
    for i in range(len(initial_centroids)):
        cluster_points = points[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=COLORS[i % len(COLORS)], s=10)

        plt.scatter(initial_centroids[:, 0],
            initial_centroids[:, 1],
            color='red',
            marker='+',
            label='Old centroid' if i == 0 else None
        )
        plt.scatter(final_centroids[:, 0],
            final_centroids[:, 1],
            color='orange',
            marker='*',
            label='New centroid' if i == 0 else None
        )

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-Means Clustering (Optimized NumPy)')
    plt.grid(True)
    plt.legend()

final_centroids, labels = k_means(rand_clst_points, centroids_cord, 500)

plot_clusters(rand_clst_points, centroids_cord, final_centroids, labels)

end_time = time.time()
print(f'Program time: {end_time - start_time:.2f} seconds')
plt.show()