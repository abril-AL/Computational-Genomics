#3.2. The String Reconstruction Problem: Q3

import sys
from typing import List, Dict, Iterable

def kmer_composition(text: str, k: int) -> Iterable[str]:
    """Forms the k-mer composition of a string."""
    return [ text[i:i+k] for i in range(0,len(text)-k+1)] 

#3.3. String Reconstruction as a Walk in the Overlap Graph: Q3, Q10

    # Q3: String Spelled by a Genome Path Problem.
def genome_path(path: List[str]) -> str:
    """Forms the genome path formed by a collection of patterns."""
    return path[0] + ''.join([elem[-1] for elem in path[1:]])
#print(genome_path(['ACCGA', 'CCGAA', 'CGAAG', 'GAAGC', 'AAGCT'])) # should return ACCGAAGCT


#3.4. Another Graph for String Reconstruction: Q6

#3.5. Walking in the de Bruijn Graph: Q8

#3.8. From Euler's Theorem to an Algorithm for Finding Eulerian Cycles: Q2, Q6, Q7