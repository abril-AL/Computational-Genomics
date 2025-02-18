# Homework 5 - Clustering Algorithms (Ch. 8)
#
# 8.6. Farther First Traversal: Q2* (recommended)
# 8.7. k-Means Clustering: Q3
# 8.8. The Lloyd Algorithm: Q3
# 8.14. Hierarchical Clustering: Q7

import sys
from typing import List, Dict, Iterable, Tuple
import math

# Farthest First Traversal
'''
FarthestFirstTraversal(Data, k) 
    Centers ← the set consisting of a single randomly chosen point from Data
    while |Centers| < k 
        DataPoint ← the point in Data maximizing d(DataPoint, Centers) 
        add DataPoint to Centers 
    return Centers '''
def farthest_first_traversal(k: int, m: int, data: List[Tuple[float, ...]]) -> List[Tuple[float, ...]]:    
    centers = set()
    centers.add(data[0])
    while centers.__len__() < k:
        dataPoint = max(data, key=lambda point: min(math.dist(point, center) for center in centers))
        centers.add(dataPoint)
    return centers

# k-Means Clustering

def euclidean_distance(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
    """Calculate the Euclidean distance between two points."""
    return sum((x - y) ** 2 for x, y in zip(p1, p2)) ** 0.5

def squared_error_distortion(k: int, m: int, centers: List[Tuple[float, ...]], data: List[Tuple[float, ...]]) -> float:
    """Calculate the squared error distortion of the data points with respect to the given centers."""
    accum = 0
    for p in data:
        close_center = min(centers, key=lambda c: euclidean_distance(c,p))
        accum += euclidean_distance(close_center,p) ** 2
    return (accum / len(data))

# The Lloyd Algorithm
import random
from typing import List, Tuple

def euclidean_distance(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
    """Calculate the Euclidean distance between two points."""
    return sum((x - y) ** 2 for x, y in zip(p1, p2)) ** 0.5

def lloyd(k: int, m: int, data: List[Tuple[float, ...]]) -> List[Tuple[float, ...]]:
    """Implement Lloyd's algorithm for k-means clustering."""
    centers = data[0:k] 
    not_converged = True
    while not_converged:
        # Assign each point to the nearest cluster center
        clusters = [[] for _ in range(k)]
        for point in data:
            closest_center_idx = min(range(k), key=lambda i: euclidean_distance(point, centers[i]))
            clusters[closest_center_idx].append(point)
        
        # Compute new centers as the mean of the clusters
        temp_centers = []
        for cluster in clusters:
            if cluster:  # Avoid division by zero
                new_center = tuple(round(sum(p[n] for p in cluster) / len(cluster)) for n in range(m))
                temp_centers.append(new_center)
            else:  # If a cluster is empty, keep its old center
                temp_centers.append(centers[len(temp_centers)])

        # Check for convergence
        if sorted(centers) == sorted(temp_centers):
            not_converged = False
        
        centers = temp_centers

    return sorted(centers)  # Ensure correct order of centers
# this works but cogniterra is annoying and wont permute answers or allow rounding by more than .001

# Hierarchical Clustering
'''HierarchicalClustering(D, n)
    Clusters ← n single-element clusters labeled 1, ... , n 
       construct a graph T with n isolated nodes labeled by single elements 1, ... , n 
    while there is more than one cluster 
        find the two closest clusters Ci and Cj 
        merge Ci and Cj into a new cluster Cnew with |Ci| + |Cj| elements
        add a new node labeled by cluster Cnew to T
        connect node Cnew to Ci and Cj by directed edges
        remove the rows and columns of D corresponding to Ci and Cj
        remove Ci and Cj from Clusters
        add a row/column to D for Cnew by computing D(Cnew, C) for each C in Clusters 
        add Cnew to Clusters 
    assign root in T as a node with no incoming edges
    return T'''
import numpy as np
from typing import List, Tuple

def hierarchical_clustering(n: int, dist: List[List[float]]) -> List[List[int]]:
    dist = np.array(dist)
    np.fill_diagonal(dist, np.inf)
    
    clusters = [[i, 1] for i in range(n)]
    newClusters = []
    adj = [[] for _ in range(n)]
    
    while len(dist) > 1:
        node_new = len(adj)
        index = np.argmin(dist)
        i = index // len(dist)
        j = index % len(dist)
        
        d_new = (dist[i, :] * clusters[i][1] + dist[j, :] * clusters[j][1]) / (clusters[i][1] + clusters[j][1])
        d_new = np.delete(d_new, [i, j], 0)
        dist = np.delete(dist, [i, j], 0)
        dist = np.delete(dist, [i, j], 1)
        dist = np.insert(dist, len(dist), d_new, 0)
        d_new = np.insert(d_new, len(d_new), np.inf, 0)
        dist = np.insert(dist, len(dist)-1, d_new, 1)
        
        adj.append([clusters[i][0], clusters[j][0]])
        clusters.append([node_new, clusters[i][1] + clusters[j][1]])
        
        if i < j:
            del clusters[j]
            del clusters[i]
        else:
            del clusters[i]
            del clusters[j]
        
        newClusters.append(find_leafs(adj, node_new))
    
    return newClusters

def find_leafs(adj: List[List[int]], v: int) -> List[int]:
    leafs = []
    visited = [False for _ in range(len(adj))]
    stack = [v]
    
    while stack:
        v = stack.pop()
        if len(adj[v]) == 0:
            leafs.append(v + 1)
        if not visited[v]:
            visited[v] = True
            stack.extend(adj[v])
    
    return leafs