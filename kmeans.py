""" KMeans class has two versions of KMeans clustering. """
from cluster import Cluster

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numpy.random import rand
from sklearn.utils.extmath import squared_norm
from munkres import Munkres, print_matrix 


class KMeans(Cluster):
    
    def __init__(self, k=5, max_iterations=100): 
        self.n_clusters = k
        self.max_iterations = max_iterations
        self.balanced = False

        
    def init_centers(self, X): 
        """ Pick up data points randomly as the initial centers of the clusters. 
        
        Parameters
        ---------------
            X : numpy array, shape (n_sample, n_features) 
                data points
                
        Returns
        ----------
            centers: numpy array, shape (n_clusters, n_features)
                initial centers of the clusters
        """
        shuffled_indices = np.random.permutation(len(X))
        center_indices = shuffled_indices[:self.n_clusters]
        centers = np.zeros(shape=(self.n_clusters, X.shape[1]), dtype=X.dtype)
        for i, idx in enumerate(center_indices): 
            centers[i] = X[idx]
            
        return centers
        
    
    def get_labels_and_inertia(self, X, centers): 
        """ Compute inertia and best labels. 
        
        Parameters
        ---------------
            X : numpy array, shape (n_sample, n_features) 
                data points
            centers : numpy array, shape (n_clusters, n_features)
                centers
            
        Returns
        ----------
           labels : numpy array, dtype=np.int, shape (n_samples,)
               Indices of clusters that samples are assigned to.
           inertia : float
               Sum of squared distances of samples to their closest cluster center.
        """
        labels = np.full(self.n_samples, -1, np.int32)
        inertia = 0.0

        # Calculate distance between each data point and each centroid. 
        for sample_idx in range(self.n_samples):
            min_dist = -1
            for center_idx in range(self.n_clusters):
                dist = 0.0

                # Get distance between the data point and the centroid. 
                # ||a - b||^2 = ||a||^2 + ||b||^2 -2<a, b>
                dist += np.dot(X[sample_idx], centers[center_idx])
                dist *= -2
                dist += np.dot(X[sample_idx], X[sample_idx])
                dist += np.dot(centers[center_idx], centers[center_idx])

                # Get minimum distance. 
                if min_dist == -1 or dist < min_dist:
                    min_dist = dist
                    # Get the cluster assigned to this datapoint. 
                    labels[sample_idx] = center_idx
                    
            # Add to inertia. 
            inertia += min_dist

        return labels, inertia
    
    
    def get_labels_and_inertia_extended(self, X, centers): 
        """ Compute inertia and best labels. 
        
        We have n_samples pre-allocated slots (roughly cluster_size = n_samples / n_clusters slots per cluster), and 
        data points can be assigned only to these slots. This will force all clusters to be of roughly same size. 
        (https://www.researchgate.net/publication/270280805_Balanced_K-Means_for_Clustering)
        
        Parameters
        ---------------
            X : numpy array, shape (n_sample, n_features) 
                data points
            centers : numpy array, shape (n_clusters, n_features)
                centers

        Returns
        ----------
           labels : numpy array, dtype=np.int, shape (n_samples,)
               Indices of clusters that samples are assigned to.
           inertia : float
               Sum of squared distances of samples to their closest cluster center.
        """
        
        # Create cost matrix. (n_samples (# of data points) x n_samples (# of slots))  
        cost_matrix = np.zeros(shape=(self.n_samples, self.n_samples), dtype=X.dtype)
        # Calculate distance between each data point and each centroid. 
        for sample_idx in range(self.n_samples):
            
            row = []
            for center_idx in range(self.n_clusters):
                dist = 0.0

                # Get distance between the data point and the centroid. 
                # ||a - b||^2 = ||a||^2 + ||b||^2 -2<a, b>
                dist += np.dot(X[sample_idx], centers[center_idx])
                dist *= -2
                dist += np.dot(X[sample_idx], X[sample_idx])
                dist += np.dot(centers[center_idx], centers[center_idx])
                
                row.append(dist)
                       
            # Make a row and add it to the matrix. 
            row = row * self.cluster_size
            row.extend([row[i] for i in range(self.n_samples % self.n_clusters)])
            cost_matrix[sample_idx] = np.array(row)
                    
        # Solve an assignment problem using Hungarian algorithm
        # to find assignment that minimizes MSE. 
        m = Munkres()
        indices = m.compute(cost_matrix)
        
        labels = np.full(self.n_samples, -1, np.int32)
        inertia = 0.0
        for row, column in indices:
            inertia += cost_matrix[row][column]
            labels[row] = column % self.n_clusters
            
        return labels, inertia
    
    
    def move_to_mean(self, X, labels): 
        """ Move centroid to the mean of the data points assigned to it. 
        
        Parameters
        ---------------
            X : numpy array, shape (n_sample, n_features) 
                data points
            labels : numpy array, dtype=np.int, shape (n_samples,)
                Indices of clusters that samples are assigned to.
            
        Returns
        ----------
           cluster_to_mean_point : numpy array, shape (n_clusters, n_features)
               new centers
        """
        cluster_to_assigned_points = dict()
        for i, cluster in enumerate(labels): 
            cluster_to_assigned_points.setdefault(cluster, []).append(X[i])

        cluster_to_mean_point =  np.zeros(shape=(self.n_clusters, self.n_features), dtype=X.dtype)
        for k, v in cluster_to_assigned_points.items():
            cluster_to_mean_point[k] = pd.Series(v).mean()

        return cluster_to_mean_point

        
    def fit(self, X): 
        
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        if self.balanced: 
            self.cluster_size = int(self.n_samples / self.n_clusters)
            
        # Place k centroids randomly. 
        centers = self.init_centers(X)
        
        best_labels, best_inertia, best_centers = None, None, None
        
        for i in range(self.max_iterations):
            centers_old = centers.copy()
            
            # Get labels and inertia. 
            if not self.balanced: 
                labels, inertia = self.get_labels_and_inertia(X, centers)
            else: 
                labels, inertia = self.get_labels_and_inertia_extended(X, centers)

            # Move the centers to the mean of the points assigned to it. 
            centers = self.move_to_mean(X, labels)
            
            print("Iteration {:2d}, inertia {:.3f}".format(i, inertia))

            # Update the labels and centers if the inertia is the minimum. 
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia

            # Check if the centers move. 
            center_shift_total = squared_norm(centers_old - centers)
            print("center shift {:f}".format(center_shift_total))          
            if center_shift_total == 0:
                print("Converged at iteration {:d}: center shift {:f}".format(i, center_shift_total))
                break
                
        # For the case it stops due to the max iterations
        if center_shift_total > 0: 
            if not self.balanced: 
                best_labels, best_inertia = self.get_labels_and_inertia(X, best_centers)
            else: 
                best_labels, best_inertia = self.get_labels_and_inertia_extended(X, best_centers)
                
        # Convert array to list for grading purpose. 
        list_best_centers = []
        for centroid in best_centers: 
            list_best_centers.append(list(centroid))
            
        return list(best_labels), list_best_centers
    
    
    def fit_extended(self, X, balanced=False): 
        
        self.balanced = balanced
        return self.fit(X)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        




