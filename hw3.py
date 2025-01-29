# 5.5 An Introduction to Dynamic Programming: The Change Problem
'''
DPChange(money, Coins)
    MinNumCoins(0) ← 0
    for m ← 1 to money
        MinNumCoins(m) ← ∞
            for i ← 0 to |Coins| - 1
                if m ≥ coini
                    if MinNumCoins(m - coini) + 1 < MinNumCoins(m)
                        MinNumCoins(m) ← MinNumCoins(m - coini) + 1
    output MinNumCoins(money)
'''
import sys
from typing import List, Dict, Iterable, Tuple
def Change(money: int, coins: List[int]) -> int:
    minNumCoins = [0] + [sys.maxsize] * money
    for m in range(1,money+1):
        minNumCoins[m] = sys.maxsize
        for i in range(0,len(coins)):
            if m >= coins[i]:
                if minNumCoins[m-coins[i]] + 1 < minNumCoins[m]:
                    minNumCoins[m] = minNumCoins[m-coins[i]] + 1
    #print(minNumCoins)
    return minNumCoins[money]
Change(40,[50,25,20,10,5,1])


import sys
from typing import List, Dict, Iterable, Tuple

'''ManhattanTourist(n, m, Down, Right)
    s0, 0 ← 0
    for i ← 1 to n
        si, 0 ← si-1, 0 + downi-1, 0
    for j ← 1 to m
        s0, j ← s0, j−1 + right0, j-1
    for i ← 1 to n
        for j ← 1 to m
            si, j ← max{si - 1, j + downi-1, j, si, j - 1 + righti, j-1}
    return sn, m'''

def longest_path_length(n: int, m: int, down: List[List[int]], right: List[List[int]]) -> int:
    s = [[ 0 for i in range(m+1)] for j in range(n+1)] # n x m matrix for partial results
    s[0][0] = 0
    for i in range(1,n+1):
        s[i][0] = s[i-1][0] + down[i-1][0]
    for j in range(1,m+1):
        s[0][j] = s[0][j-1] + right[0][j-1]
    for i in range(1,n+1):
        for j in range(1,m+1):
            s[i][j] = max(s[i-1][j] + down[i-1][j], s[i][j-1] + right[i][j-1])
    #print(s[n][m])
    return s[n][m]


longest_path_length(4,4, [[1, 0, 2, 4, 3], [4, 6, 5 ,2, 1],[4, 4, 5, 2, 1],[5, 6, 8, 5, 3]],
[[ 3, 2, 4, 0],[3, 2, 4, 2],[0, 7, 3, 3],[3, 3, 0, 2],[1, 3, 2, 2]])


'''LCSBackTrack(v, w)
    for i ← 0 to |v|
        si, 0 ← 0
    for j ← 0 to |w| 
        s0, j ← 0
    for i ← 1 to |v|
        for j ← 1 to |w|
            match ← 0
            if vi-1 = wj-1
                match ← 1
            si, j ← max{si-1, j , si,j-1 , si-1, j-1 + match }
            if si,j = si-1,j
                Backtracki, j ← "↓"
            else if si, j = si, j-1
                Backtracki, j ← "→"
            else if si, j = si-1, j-1 + match
                Backtracki, j ← "↘"
    return Backtrack
    
    OutputLCS(backtrack, v, i, j)
    if i = 0 or j = 0
        return ""
    if backtracki, j = "↓"
        return OutputLCS(backtrack, v, i - 1, j)
    else if backtracki, j = "→"
        return OutputLCS(backtrack, v, i, j - 1)
    else
        return OutputLCS(backtrack, v, i - 1, j - 1) + vi    '''
import sys
from typing import List, Dict, Iterable, Tuple
sys.setrecursionlimit(10000)

def LCSBacktrack(v,w):
    backtrack = [[0 for i in range(len(w)+1)] for j in range(len(v)+1)]
    s = [[0 for i in range(len(w)+1)] for j in range(len(v)+1)]
    for i in range(1,len(v)+1):
        s[i][0] = 0
    for j in range(1,len(w)+1):
        s[0][j] = 0
    for i in range(1,len(v)+1):
        for j in range(1,len(w)+1):
            match = 0
            if v[i-1] == w[j-1]:
                match = 1
            s[i][j] = max(s[i-1][j],s[i][j-1],s[i-1][j-1]+match)   
            if s[i][j] == s[i-1][j]:
                backtrack[i][j] = "↓"
            elif s[i][j] == s[i][j-1]:
                backtrack[i][j] = "→"
            elif s[i][j] == s[i-1][j-1] + match:
                backtrack[i][j] = "↘"
    return backtrack

def OutputLCS(backtrack,v,i,j):
    if i == 0 or j == 0:
        return ""
    if backtrack[i][j] == "↓":
        return OutputLCS(backtrack,v,i-1,j)
    elif backtrack[i][j] == "→":
        return OutputLCS(backtrack,v,i,j-1)
    else:
        return OutputLCS(backtrack,v,i-1,j-1) + v[i-1]


def longest_common_subsequence(s: str, t: str) -> str:
    backtrack = LCSBacktrack(s,t)
    return OutputLCS(backtrack,s,len(s),len(t))

#print(longest_common_subsequence("AACCTTGG","ACACTGTGA"))
#print(longest_common_subsequence("GACT","ATG"))


import sys
from typing import List, Dict, Tuple

def topological_ordering(e: Dict[int, List[Tuple[int, int]]]) -> List[int]:
    """
    Perform a topological sort on a directed acyclic graph.

    Args:
    e (Dict[int, List[Tuple[int, int]]]): A dictionary representing the graph, where keys are nodes,
        and values are lists of tuples (neighbor, weight).

    Returns:
    List[int]: A list of nodes in topologically sorted order.
    """
    visited = set()
    stack = []

    def visit(node):
        if node not in visited:
            visited.add(node)
            for neighbor, _ in e.get(node, []):
                visit(neighbor)
            stack.append(node)

    # Ensure all nodes are visited, even if they have no outgoing edges
    all_nodes = set(e.keys())
    for neighbors in e.values():
        for neighbor, _ in neighbors:
            all_nodes.add(neighbor)

    for node in all_nodes:
        visit(node)

    stack.reverse()
    return stack

def longest_path(s: int, t: int, e: Dict[int, List[Tuple[int, int]]]) -> Tuple[int, List[int]]:
    """
    Calculate the longest path between two nodes in a weighted directed acyclic graph.

    Args:
    s (int): The starting node.
    t (int): The ending node.
    e (Dict[int, List[Tuple[int, int]]]): A dictionary representing the graph, where keys are nodes,
        and values are lists of tuples (neighbor, weight).

    Returns:
    Tuple[int, List[int]]: A tuple containing the length of the longest path and the list of nodes in the path.
    """
    # Topological ordering
    ordering = topological_ordering(e)
    
    # Initialize the longest path and the predecessor
    longest = {node: -sys.maxsize for node in ordering}
    predecessor = {node: None for node in ordering}
    longest[s] = 0
    
    # Traverse the nodes in topological order
    for node in ordering:
        if longest[node] != -sys.maxsize:  # Only process reachable nodes
            for neighbor, weight in e.get(node, []):
                if longest[node] + weight > longest[neighbor]:
                    longest[neighbor] = longest[node] + weight
                    predecessor[neighbor] = node
    
    # If the target node is unreachable, return an appropriate value
    if longest[t] == -sys.maxsize:
        return -1, []  # Indicates no path exists
    
    # Reconstruct the path
    path = []
    node = t
    while node is not None:
        path.append(node)
        node = predecessor[node]
    path.reverse()
    
    return longest[t], path

# Example usage:
graph = {
    0: [(1, 5), (2, 3)],
    1: [(3, 6)],
    2: [(3, 7)],
    3: []
}
print(longest_path(0, 3, graph))