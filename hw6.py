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

'''GreedyMotifSearch(Dna, k, t)
    BestMotifs ← motif matrix formed by first k-mers in each string from Dna
    for each k-mer Motif in the first string from Dna
        Motif1 ← Motif
        for i = 2 to t
            form Profile from motifs Motif1, …, Motifi - 1
            Motifi ← Profile-most probable k-mer in the i-th string in Dna
        Motifs ← (Motif1, …, Motift)
        if Score(Motifs) < Score(BestMotifs)
            BestMotifs ← Motifs
    return BestMotifs'''

# Greedy Motif Search
def greedy_motif_search(dna: list[str], k: int, t: int) -> list[str]:
    BestMotifs = []
    for i in range(0, t):
        BestMotifs.append(dna[i][0:k])
    n = len(dna[0])
    for i in range(n-k+1):
        Motifs = []
        Motifs.append(dna[0][i:i+k])
        for j in range(1, t):
            P = build_profile(Motifs[0:j])
            Motifs.append(ProfileMostProbablePattern(dna[j], k, P))
        if Score(Motifs) < Score(BestMotifs):
            BestMotifs = Motifs
    return BestMotifs

def build_profile(Motifs):
    count = {} # initializing the count dictionary
    profile = {}
    k = len(Motifs[0])
    for symbol in "ACGT":
        count[symbol] = []
        for j in range(k):
            count[symbol].append(0)

    t = len(Motifs)
    for i in range(t):
        for j in range(k):
            symbol = Motifs[i][j]
            count[symbol][j] += 1
    ## divide the number of motif strings to get frequency
    for letter in count.keys():
        profile[letter] = [x/ float(t) for x in count[letter]]
    return profile

def Score(motifs):
    """Calculates the score of a motif matrix based on the number of non-consensus nucleotides."""
    k = len(motifs[0])
    consensus = ""
    score = 0
    
    for i in range(k):
        column = [motif[i] for motif in motifs]
        most_common = max(set(column), key=column.count)
        consensus += most_common
        score += sum(1 for char in column if char != most_common)
    
    return score

def ProfileMostProbablePattern(Text, k, Profile):
    p_dict = {}
    for i in range(len(Text)- k +1):
        p = Pr(Text[i: i+k], Profile)
        p_dict[i] = p
    m = max(p_dict.values())
    keys = [k for k,v in p_dict.items() if v == m]
    ind = keys[0]
    return Text[ind: ind +k]

def Pr(Text, Profile):
    p = 1
    for i in range(len(Text)):
        p = p * Profile[Text[i]][i]
    return p