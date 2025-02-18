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