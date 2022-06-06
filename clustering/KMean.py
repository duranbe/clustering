import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


class KMean:
    def __init__(self, x, k, distance="euclidean"):

        self.x = x  # Input Points
        self.k = k  # Nb of Clusters
        self.dim = x.shape[-1]  # Point Dimension
        self.n = x.shape[0]  # Nb of Points
        self.distance = distance  # Method to use for distance calculation
        self.belongs = None # n value array which contain for each point the id of the cluster
       
        # Generating random points for cluster's positions
        self.middles = np.random.randint(low=0, high=20, size=(k, self.dim)) / 2

        self.code_color = [
            "red",
            "green",
            "blue",
            "black",
            "yellow",
            "orange",
            "purple",
            "brown",
            "grey",
        ]
        self.colors = []
        self.colormap = {}

        self.distance_function = {
            "euclidean": self._euclidean_distance,
            "manhattan": self._manhattan_distance,
            "cosine": self._cosine_distance
        }

    def _euclidean_distance(self):

        k = self.k
        n = self.n
        x = self.x
        l = self.dim
        middles = self.middles

        # (k,n,l)
        points = np.broadcast_to(x, (k, n, l))

        c = np.zeros((k, n, l))

        # Need to find a numpy way to do this to avoid loops
        # (k,n,l)
        for i in range(0, k):
            c[i] = np.broadcast_to(middles[i], (n, l))

            # Euclidean Distance
            # (k,n) because distance is a scalar here
        d = np.linalg.norm(points - c, axis=2)

        # For each point get the argument of the minimum distance
        args = np.argmin(d, axis=0)

        return args

    def _manhattan_distance(self):
        k = self.k
        n = self.n
        x = self.x
        l = self.dim
        middles = self.middles

        # (k,n,l)
        points = np.broadcast_to(x, (k, n, l))

        c = np.zeros((k, n, l))

        # Need to find a numpy way to do this to avoid loops
        # (k,n,l)
        for i in range(0, k):
            c[i] = np.broadcast_to(middles[i], (n, l))

            # Manhattan Distance
            # (k,n) because distance is a scalar here
        d = np.sum(abs(points - c), axis=2)

        # For each point get the argument of the minimum distance
        args = np.argmin(d, axis=0)

        return args

    def _cosine_distance(self):
        k = self.k
        n = self.n
        x = self.x
        l = self.dim
        middles = self.middles

        # (k,n,l)
        points = np.broadcast_to(x, (k, n, l))

        c = np.zeros((k, n, l))

        # Need to find a numpy way to do this to avoid loops
        # (k,n,l)
        for i in range(0, k):
            c[i] = np.broadcast_to(middles[i], (n, l))

            # Cosine Distance
            # (k,n) because distance is a scalar here
        d = np.zeros((k, n))
        for i in range(0, k):
            for j in range(0, n):
                d[i][j] = scipy.spatial.distance.cosine(
                    points[i][j], c[i][j]
                )  # cosine distance

                # For each point get the argument of the minimum distance
        args = np.argmin(d, axis=0)

        return args

    def _scatter_plot(self):
        k = self.k
        n = self.n
        points = self.x
        l = self.dim
        middles = self.middles

        code_color = self.code_color
        colors = self.colors
        colormap = self.colormap

        for i in range(k):
            colormap[i] = code_color[i]
            colors.append(code_color[i])

        plt.scatter(points[:, 0], points[:, 1], color="blue")
        plt.scatter(middles[:, 0], middles[:, 1], color=colors)
        for i in range(0, n):
            plt.scatter(points[i][0], points[i][1], color=colormap[self.belongs[i]])
            plt.text(points[i][0], points[i][1], str(i), fontsize=15)
        for i in range(0, k):
            plt.text(middles[i, 0], middles[i, 1], chr(i + 65), fontsize=15)

    def _new_middles(self):

        # Calcul of newer positions for clusters using the mean coordinates

        k = self.k
        n = self.n
        l = self.dim

        middles = self.middles

        new_middles = np.zeros((k, l))

        for i in range(0, k):
            if i in self.belongs:
                new_middles[i] = np.mean(self.x[self.belongs == i, :], axis=0)
            else:
                new_middles[i] = middles[i]

        return new_middles

    def run(self, n_iter=5):
        # Running the KMean algorithm

        dist_function = self.distance_function[self.distance]

        for i in range(n_iter):
            self.belongs = dist_function()
            self.middles = self._new_middles()
        
        self._scatter_plot()
        plt.show()
