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

#0: 3
#1: 0
#2: 1 6
#3: 2
#4: 2
#5: 4
#6: 5 8
#7: 9
#8: 7
#9: 6
# 0 3 2 6 8 7 9 6 5 4 2 1 0

    # Q6: Eulerian Path Problem.
from typing import List, Dict, Iterable
from collections import defaultdict
import random
from copy import deepcopy
from collections import Counter
'''
def eulerian_path(adj_dict):

    # Adjacency dictionary with 'false' edge added to make a cycle
    cycle_d = deepcopy(adj_dict)

    # Initializes Counter that will track which nodes are unbalanced,
    # which gives us the start and finish
    find_ends = Counter()

    # Go through each node, using the counter find_ends to count how many nodes
    # are adjacent to the current node and subtract 1 for each of those adj
    for row in cycle_d.items():
        dir_out = row[0]
        dir_ins = row[1]

        for ins in dir_ins:
            find_ends[ins] -= 1
            find_ends[dir_out] += 1

    # The most_common function return a list of elements and their counts
    # from the most common to the least.
    # The starting node (V0) will be most common, as there is 1 less node
    # adjacent to it, while the ending node will have 1 less adjacency
    start = find_ends.most_common()[0][0]
    end = find_ends.most_common()[-1][0]

    # Create a 'false' edge between the end node and the start node so as to
    # enable the algorithm for Eulerian cycles to function
    try:
        cycle_d[end].append(start)
    except KeyError:
        cycle_d[end] = [start]

    cycle = eulerian_cycle(cycle_d)

    # Eulerian cycle needs to be re-oriented to begin with the "true" starteulerian_
    for i, n in enumerate(cycle):
        if n == end:
            if cycle[i+1] == start:
                break_point = i
                break

    cycle =  cycle[break_point+1:] + cycle[1:break_point+1]

    return cycle

def eulerian_cycle(adj_dict):

    # Dictionary that tracks remaining edges (those not yet taken),
    # initialized as the input adjacency dict
    remain_d = deepcopy(adj_dict)

    # Randomly select a starting node
    node = random.choice(range(len(adj_dict)))

    # Create list to track Eulerian cycle
    cycle= [node]

    # Continue as long as there are edges that remain untaken
    while len(remain_d) > 0:

        # Check if any adjacencies remain for the node and if so, how many
        value = remain_d.get(node)

        # If the node has no unused edges, we must be back at V0 (since the
        # graph is balanced), but we also know there are remaining edges out
        # there. We need to expand the circle until it encompasses all nodes.
        if value == None:

            # To do so, iterate through the current "cycle" until we find a
            # node with an unused edge. Make that the new V0.
            for i, n in enumerate(cycle):
                if remain_d.get(n) > 0:
                    node = n
                    cycle = cycle[i:]+cycle[1:i+1]
                    break

        # If the node has a single unused edge, simply use it to continue the
        # cycle by adding it to the list 'cycle' and removing it from the dict
        elif len(value) == 1:
            node = remain_d.pop(node)[0]
            cycle.append(node)

        # If the node has multiple unused edges, randomly select one to add to
        # the cycle and remove it out from the node's list of adjacencies.
        elif len(value) > 1:
            random_i = random.randrange(len(value))
            pos_nodes = remain_d[node]
            new_node = pos_nodes.pop(random_i)
            remain_d[node] = pos_nodes
            node = new_node
            cycle.append(node)

    return cycle
'''