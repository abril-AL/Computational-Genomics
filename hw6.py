#    HW6 – Chapter 2 (due Tue Feb 25th, 2025 at 12pm)       #
#############################################################
# 2.2. Motif Finding Is More Difficult Than You Think: Q8 
# 2.5. Greedy Motif Search: Q2, Q3 
# 2.6. Motif Finding Meets Oliver Cromwell Q9* (recommended) 
# 2.7. Randomized Motif Search: Q5* (recommended) 
# 2.9. Gibbs Sampling: Q11* (recommended)

# Implanted Motif Problem: Find all (k, d)-motifs in a collection of strings.
'''MotifEnumeration(Dna, k, d)
    Patterns ← an empty set
    for each k-mer Pattern in Dna
        for each k-mer Pattern’ differing from Pattern by at most d mismatches
            if Pattern' appears in each string from Dna with at most d mismatches
                add Pattern' to Patterns
    remove duplicates from Patterns
    return Patterns'''
import sys
from hw1 import hamming_distance 
from hw1 import neighbors as generate_neighbors
def motif_enumeration(dna: list[str], k: int, d: int) -> list[str]:
    """Implements the MotifEnumeration algorithm."""
    patterns = set()
    for sequence in dna:
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i + k]
            neighbors = generate_neighbors(kmer, d)
            
            for neighbor in neighbors:
                if all(any(hamming_distance(neighbor, sequence[j:j + k]) <= d for j in range(len(sequence) - k + 1)) for sequence in dna):
                    patterns.add(neighbor)
    return sorted(patterns)
#r = motif_enumeration(['ATTTGGC','TGCCTTA','CGGTATC','GAAAATT'],3,1)
#print(r)

# Profile-most Probable k-mer Problem
def profile_most_probable_kmer(text: str, k: int,
                               profile: list[dict[str, float]]) -> str:
    """Identifies the most probable k-mer according to a given profile matrix.

    The profile matrix is represented as a list of columns, where the i-th element is a map
    whose keys are strings ("A", "C", "G", and "T") and whose values represent the probability
    associated with this symbol in the i-th column of the profile matrix.
    """
    probs = [] 
    for window in [text[i:i+k] for i in range(len(text)-k+1)]:
        prob = 1
        for (n,i) in zip(window,range(len(window))):
            prob = prob * profile[i][n]
        probs.append(prob)
    max_prob_index = probs.index(max(probs))
    return text[max_prob_index:max_prob_index + k]