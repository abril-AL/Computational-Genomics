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

print(de_bruijn_kmers(["GAGG", "CAGG", "GGGG", "GGGA", "CAGG", "AGGG", "GGAG"]))


#GAGG CAGG GGGG GGGA CAGG AGGG GGAG
#AGG: GGG
#CAG: AGG AGG
#GAG: AGG
#GGA: GAG
#GGG: GGA GGG

#3.8. From Euler's Theorem to an Algorithm for Finding Eulerian Cycles: Q2, Q6, Q7