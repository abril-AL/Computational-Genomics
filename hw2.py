##########################
#      Homework 2        #
#                        #
#   9.3, 9.5, 9.6, 9.7   #
#                        #      
##########################

# 9.3 Herding Patterns into a Trie


# Code Challenge: Solve the Trie Construction Problem.
'''
TrieConstruction(Patterns)
    Trie ← a graph consisting of a single node root
    for each string Pattern in Patterns
        currentNode ← root
        for i ← 0 to |Pattern| - 1
            currentSymbol ← Pattern[i]
            if there is an outgoing edge from currentNode with label currentSymbol
                currentNode ← ending node of this edge
            else
                add a new node newNode to Trie
                add a new edge from currentNode to newNode with label currentSymbol
                currentNode ← newNode
    return Trie'''

import sys
from typing import List, Dict, Iterable, Tuple, Set

# Insert your trie_construction function here, along with any subroutines you need
def trie_construction(patterns: List[str]) -> List[Tuple[int, int, str]]:
    """Construct a trie from a collection of patterns."""
    trie = [] 
    for pattern in patterns:
        #print(pattern)
        currNode = 0 # root    
        for i in range(0,len(pattern)):
            currSymbol = pattern[i] # char 
            if currSymbol in [tup[2] for tup in trie if tup[0] == currNode]:
                currNode = [ tup[1] for tup in trie if (tup[0] == currNode and tup[2] == currSymbol) ]
                #print(currNode[0],currSymbol,"found", currNode)
                currNode = currNode[0]
            else:
                ind = len(trie) + 1
                #print('adding',ind,currNode,currSymbol)
                trie.append(  (currNode, ind, currSymbol) )
                currNode = ind
    trie.sort(key=lambda x: x[0])
    #print(trie)
    return trie

#trie_construction(["ATAGA","ATC","GAT"])

'''
PrefixTrieMatching(Text, Trie)
    symbol ← first letter of Text
    v ← root of Trie
    while forever
        if v is a leaf in Trie
            output the pattern spelled by the path from the root to v
        else if there is an edge (v, w) in Trie labeled by symbol
            symbol ← next letter of Text
            v ← w
        else
            return "no matches found"

TrieMatching(Text, Trie)
    while Text is nonempty
        PrefixTrieMatching(Text, Trie)
        remove first symbol from Text
'''

def prefixMathcing(text,tria):
    symbol_index = 0
    v = 0 # root
    while True:
        out_edges = [tup for tup in tria if tup[0] == v]
        #print('CL:',check_leaf)
        if not out_edges:  # is leaf - no outgoing edges
            return True
        if symbol_index >= len(text):
            return False
        
        symbol = text[symbol_index]
        matching_edges = [tup for tup in out_edges if tup[2] == symbol]
        if matching_edges:
            symbol_index += 1
            v = matching_edges[0][1]
        else:
            return False

def trie_matching(text: str, patterns: List[str]) -> Dict[str, List[int]]:
    '''Find all starting positions in Text where a string from Patterns appears as a substring.'''
    t = trie_construction(patterns)
    #print(t)
    p = [i for i in range(0,len(text)) if prefixMathcing(text[i:],t)]
    res = {pattern: [i for i in p if text[i:i+len(pattern)] == pattern] for pattern in patterns}
    #print(res)
    return res

#print(prefixMathcing("AB", [(0, 1, 'A'), (1, 2, 'B'), (0,3,'C')]))
trie_matching( "AATCGGGTTCAATCGGGGT", ["ATCG"] )


# 9.5 Suffix Trees
 

#Code Challenge: Solve the Suffix Tree Construction Problem.


'''
ModifiedSuffixTrieConstruction(Text)
    Trie ← a graph consisting of a single node root
    for i ← 0 to |Text| - 1
        currentNode ← root
        for j ← i to |Text| - 1
            currentSymbol ← j-th symbol of Text
            if there is an outgoing edge from currentNode labeled by currentSymbol
                currentNode ← ending node of this edge
            else
                add a new node newNode to Trie
                add an edge newEdge connecting currentNode to newNode in Trie
                Symbol(newEdge) ← currentSymbol
                Position(newEdge) ← j
                currentNode ← newNode
        if currentNode is a leaf in Trie
            assign label i to this leaf
    return Trie
'''

'''
The following pseudocode constructs a suffix tree using the modified suffix trie constructed by ModifiedSuffixTrieConstruction(). 
This algorithm will consolidate each non-branching path of the modified suffix trie into a single edge. 
For more on non-branching paths, recall Charging Station: Maximal Non-Branching Paths in a Graph.

ModifiedSuffixTreeConstruction(Text)
    Trie ← ModifiedSuffixTrieConstruction
    for each non-branching path Path in Trie
        substitute Path by a single edge e connecting the first and last nodes of Path
        Position(e) ← Position(first edge of Path)
        Length(e) ← number of edges of Path
    return Trie
'''

