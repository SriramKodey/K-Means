import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk_met
import math
import random

class kmeans_cluster:
    def __init__(self, data, k):
        self.k = k
        self.data = data
        self.n_pts = data.shape[0]
        self.vec_length = data.shape[1]
        self.dist_to_cent = np.empty([self.k, self.n_pts])
        self.centroids = np.empty([self.k, self.vec_length])
        self.labels = np.empty(self.n_pts, dtype=np.uint8)

    def get_distance(self, arr1, arr2):
        dist = 0

        for i in range(len(arr1)):
            dist += pow((arr1[i] - arr2[i]), 2)

        dist = math.sqrt(dist)
        return dist

    def generate_dist_to_cent(self):
        for i in range(self.k):
            for j in range(self.n_pts):
                self.dist_to_cent[i][j] = self.get_distance(self.data[j], self.centroids[i])


    def random_cent(self):
        nums = range(self.n_pts)
        c = random.sample(nums, self.k)
        
        centroids = np.empty([self.k, self.vec_length])
        for i in range(self.k):
            centroids[i] = self.data[c[i]]

        return centroids
        

    def cluster(self):
        # Set random centroids for the first step.
        self.centroids = self.random_cent()

        flag = True
        while flag:
            self.generate_dist_to_cent()
            label_size = np.zeros(self.k)
            new_centroids = np.zeros([self.k, self.vec_length])

            for i in range(self.n_pts):
                arr = self.dist_to_cent[:, i]
                label = np.argmin(arr)
                self.labels[i] = label
                label_size[label] += 1
                new_centroids[label] = np.add(new_centroids[label], self.data[i])

            for i in range(self.k):
                new_centroids[i] = new_centroids[i]/label_size[i]


            if (np.array_equal(self.centroids, new_centroids)):
                flag = False

            else:
                self.centroids = new_centroids   


    def metrics(self):
        silhouette = sk_met.silhouette_score(self.data, self.labels, metric = 'euclidean', sample_size=None, random_state=None)
        db_score = sk_met.davies_bouldin_score(self.data, self.labels)
        print("The Silhouette Score is = ", silhouette)
        print("The Davies Bouldin Score is = ", db_score)


    
    def plot(self):
        print(self.labels)
        


