import sys
from typing import List, Dict, Iterable

#3.2. The String Reconstruction Problem: Q3

def kmer_composition(text: str, k: int) -> Iterable[str]:
    """Forms the k-mer composition of a string."""
    return [ text[i:i+k] for i in range(0,len(text)-k+1)] 

#3.3. String Reconstruction as a Walk in the Overlap Graph: Q3, Q10

    # Q3: String Spelled by a Genome Path Problem.
def genome_path(path: List[str]) -> str:
    """Forms the genome path formed by a collection of patterns."""
    return path[0] + ''.join([elem[-1] for elem in path[1:]])
#print(genome_path(['ACCGA', 'CCGAA', 'CGAAG', 'GAAGC', 'AAGCT'])) # should return ACCGAAGCT

    # Q10: Overlap Graph Problem 
def overlap_graph(patterns: List[str]) -> Dict[str, List[str]]:
    """Forms the overlap graph of a collection of patterns."""
    res = {}
    for p in patterns: 
        s = list(set([nxt for nxt in patterns if  (p[1:] == nxt[:-1])])) # (nxt is not p) and
        if s:
            res[p] = s
    return res
#print(overlap_graph(['AAG', 'AGA', 'ATT', 'CTA', 'CTC', 'GAT', 'TAC', "TCT", "TCT", "TTC"]))

#3.4. Another Graph for String Reconstruction: Q6

    # Q6: De Bruijn Graph from a String Problem
def de_bruijn_string(text: str, k: int) -> Dict[str, List[str]]:
    """Forms the de Bruijn graph of a string."""    
    adj_list = {} # str: [ ..., ..., ...  ] 
    
    for i in range(0,len(text)-k+1):
        kmer = text[i:i+k]
        pre = kmer[:-1]
        suf = kmer[1:]

        if pre not in adj_list:
            adj_list[pre] = []
        adj_list[pre].append(suf) 
    
    return adj_list

#print(de_bruijn_string("ACGTGTATA",3))

#ACGTGTATA =>
#AC: CG
#AT: TA
#CG: GT
#GT: TA TG
#TA: AT
#TG: GT

#3.5. Walking in the de Bruijn Graph: Q8
        
    # Q8: Construct the de Bruijn graph from a set of k-mers.
def de_bruijn_kmers(k_mers: List[str]) -> Dict[str, List[str]]:
    """Forms the de Bruijn graph of a collection of k-mers."""
    k = len(k_mers[0])
    res = {}

    for kmer in k_mers:
        pre = kmer[:-1]
        suf = kmer[1:]

        if pre not in res:
            res[pre] = []
        res[pre].append(suf)
        
    return res

#print(de_bruijn_kmers(["GAGG", "CAGG", "GGGG", "GGGA", "CAGG", "AGGG", "GGAG"]))

#GAGG CAGG GGGG GGGA CAGG AGGG GGAG
#AGG: GGG
#CAG: AGG AGG
#GAG: AGG
#GGA: GAG
#GGG: GGA GGG

#3.8. From Euler's Theorem to an Algorithm for Finding Eulerian Cycles: Q2, Q6, Q7

# Q2: Eulerian Cycle Problem
'''EulerianCycle(Graph)
    form a cycle Cycle by randomly walking in Graph (don't visit the same edge twice!)
    while there are unexplored edges in Graph
        select a node newStart in Cycle with still unexplored edges
        form Cycle’ by traversing Cycle (starting at newStart) and then randomly walking 
        Cycle ← Cycle’
    return Cycle'''

# g[u] is the list of neighbors of the vertex u
def eulerian_cycle(g: Dict[int, List[int]]) -> Iterable[int]:
    """Constructs an Eulerian cycle in a graph."""
    graph = {u: list(neighbors) for u, neighbors in g.items()}
    #print("G:",graph)
    
    stack = [next(iter(graph))]  # Start with the first vertex
    print("s:",stack)
    cycle = []

    while stack:
        u = stack[-1]
        if graph[u]:  # if unvisited edges from u
            v = graph[u].pop()  # walk the edge u -> v
            stack.append(v)
        else:
            cycle.append(stack.pop())  # add to cylcle

    return cycle[::-1]
    
graph = {
    0: [3],
    1: [0],
    2: [1, 6],
    3: [2],
    4: [2],
    5: [4],
    6: [5, 8],
    7: [9],
    8: [7],
    9: [6]
}

#print(eulerian_cycle(graph))

    # Q6: Eulerian Path Problem.
from typing import List, Dict, Iterable
from collections import defaultdict
import random
from copy import deepcopy
from collections import Counter

g2 = {
    0: [2],
    1: [3],
    2: [1],
    3: [0, 4],
    6: [3, 7],
    7: [8],
    8: [9],
    9: [6]
}
# 6 7 8 9 6 3 0 2 1 3 4

def eulerian_path(adj_dict):
    in_deg = Counter()
    out_deg = Counter()

    for node in adj_dict:
        out_deg[node] += len(adj_dict[node])
        for neighbor in adj_dict[node]:
            in_deg[neighbor] += 1
    
    #print("Out:",out_deg)
    #print("In:",in_deg)

    start_node = None
    end_node = None
    for node in set(list(in_deg.keys()) + list(out_deg.keys())):
        in_d = in_deg.get(node, 0)
        out_d = out_deg.get(node, 0)
        if in_d - out_d == -1:
            start_node = node
        elif in_d - out_d == 1:
            end_node = node
        elif in_d - out_d != 0:
            raise Exception("Graph is not Eulerian")

    #print("Start node:", start_node) # 6
    #print("End node:", end_node) # 4

    # construct Eulerian cycle 
    graph = {u: list(adj_dict[u]) for u in adj_dict}
    for node in in_deg.keys():
        if node not in graph:
            graph[node] = [] 
    
    stack = [start_node]  # Start with the start vertex
    path = []

    while stack:
        #print('Stack:', stack)
        #print('Current Node:', stack[-1])
        if stack[-1] not in graph:
            raise KeyError(f"Node {stack[-1]} is missing from graph")
        #print('Edges:', graph[stack[-1]])  
        while graph[stack[-1]]:  # This is where the KeyError likely occurs
            stack.append(graph[stack[-1]].pop())
        path.append(stack.pop())

    return path[::-1]

print(eulerian_path(g2))