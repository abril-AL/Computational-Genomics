###############
# CS 122 HW 1 #
###############

import sys
# 1.2 Hidden Messages in the Replication Origin
def frequent_words(text: str, k: int) -> list[str]:
    # Given text and len of k-mer, return lst of freq k-mers
    
    # Build map of k-mer patters and freq
    dict = {}
    for i in range(0,len(text) - k +1):
        sub = text[i:i+k]
        if sub in dict:
            dict[sub] += 1
        else:
            dict[sub] = 1
    
    # Find max cnt
    max = 0
    for kmer in dict:
        if dict[kmer] > max:
            max = dict[kmer]
            
    # Add each kmer w that count to res
    res = []
    for kmer in dict:
        if dict[kmer] is max:
            res.append(kmer)
    
    return res


# 1.3 Some Hidden Messages are More Surprising than Others
# Insert your reverse_complement function here, along with any subroutines you need
def reverse_complement(pattern: str) -> str:
    """Calculate the reverse complement of a DNA pattern."""
    stack = [] 
    for s in pattern:
        stack.append(help(s))
    res = []
    for i in range(len(stack)):
        res.append(stack.pop())
    return ''.join(res)
    
def help(s):
    if s == "A": return "T"
    elif s == "T": return "A"
    elif s == "G": return "C"
    else: return "G"
#print(reverse_complement("AAAACCCGGT"))# ACCGGGTTTT

def pattern_matching(pattern: str, genome: str) -> list[int]:
    """Find all occurrences (index) of a pattern in a genome."""
    res = []
    for i in range(len(genome) - len(pattern) + 1 ):
        if genome[i:i+len(pattern)] == pattern:
            res.append(i)
    return res

f = open("Vibrio_cholerae.txt", "r")
genome = f.read()
res = pattern_matching("CTTGATCAT", genome)
#for i in res:
#    print(i, end = " ")
    
# 1.4 Clump Finding Problem: Find patterns forming clumps in a string.

# Input: A string Genome, and integers k, L, and t.
# Output: All distinct k-mers forming (L, t)-clumps in Genome.

''''
FindClumps(Text, k, L, t)
    Patterns ← an array of strings of length 0
    n ← |Text|
    for every integer i between 0 and n − L
        Window ← Text(i, L)
        freqMap ← FrequencyTable(Window, k)
        for every key s in freqMap
            if freqMap[s] ≥ t
                append s to Patterns
    remove duplicates from Patterns
    return Patterns
''' 
def find_clumps(genome: str, k: int, l: int, t: int) -> list[str]:
    patterns = []
    n = len(genome)
    for i in range(0,n-l+1):
        win = genome[i:i+l]
        freqMap = frequency_table(win, k)
        for s in freqMap:
            if freqMap[s] >= t:
                patterns.append(s)
    return list(set(patterns))

def frequency_table(s,k):
    freqMap = {}
    for i in range(len(s) - k + 1):
        pattern = s[i:i+k]
        if pattern in freqMap:
            freqMap[pattern] += 1
        else:
            freqMap[pattern] = 1
    return freqMap 
#print(find_clumps("CGGACTCGACAGATGTGAAGAACGACAATGTGAAGACTCGACACGACAGAGTGAAGAGAAGAGGAAACATTGTAA", 5, 50, 4)) # K L t)
# GAAGA CGACA

# 1.6 Asymmetry of Replication
'''
c-> dec by 1
g-> inc by 1
a or t -> no change
'''
def minimum_skew(genome: str) -> list[int]:
	s= 0
	min_skew = 0
	skew_list = []
	index = 0
	for i in genome:
		index += 1
		if i == 'C':
			s -= 1
		elif i == 'G':
			s += 1
		if s < min_skew:
			skew_list = [index]
			min_skew = s
		if s == min_skew and index not in skew_list:
			skew_list.append(index)	
	return skew_list

#print(minimum_skew("TAAAGACTGCCGAGAGGCCAACACGAGTGCTAGAACGAGGGGCGTAAACGCGGGTCCGAT"))

# Hamming Distance Problem
def hamming_distance(p: str, q: str) -> int:
    count = 0
    for pi,qi in zip(p,q):
        if pi != qi:
            count += 1
    return count

def approximate_pattern_matching(pattern: str, text: str, d: int) -> list[int]:
    res = []
    for i in range(len(text)-len(pattern)+1):
        if hamming_distance(pattern,text[i:len(pattern)+i]) <= d:
            res.append(i)
    return res
         
# Count_2(AACAAGCTGATAAACATTTAAAGAG, AAAAA).
#print(len(approximate_pattern_matching("AAAAA","AACAAGCTGATAAACATTTAAAGAG",2)))

''' ApproximatePatternCount(Text, Pattern, d)
    count ← 0
    for i ← 0 to |Text| − |Pattern|
        Pattern′ ← Text(i , |Pattern|)
        if HammingDistance(Pattern, Pattern′) ≤ d
            count ← count + 1
    return count    '''

def approximate_pattern_count(text: str, pattern: str, d: int) -> int:
    count = 0
    for i in range(0,len(text) - len(pattern)+1):
         if hamming_distance(pattern, text[i:i+len(pattern)]) <= d:
             count += 1
    return count


# 1.11 Generating the Neighborhood of a String
'''
Neighbors(Pattern, d)
    if d = 0
        return {Pattern}
    if |Pattern| = 1
        return {A, C, G, T}
    Neighborhood ← an empty set
    SuffixNeighbors ← Neighbors(Suffix(Pattern), d)
    for each string Text from SuffixNeighbors
        if HammingDistance(Suffix(Pattern), Text) < d
            for each nucleotide x
                add x • Text to Neighborhood
        else
            add FirstSymbol(Pattern) • Text to Neighborhood
    return Neighborhood'''

def neighbors(s: str, d: int) -> list[str]:
    if d == 0: return [s]
    if len(s) == 1: return {'A','C','G','T'}
    nh = []
    suff_ns = neighbors(s[1:], d)
    for text in suff_ns:
        if hamming_distance(s[1:], text) < d:
            for x in ['A','C','G','T']:
                nh.append(x + text)
        else:
            nh.append(s[0] + text)
    return nh
#print(Neighbors("AAAAA",1))

# 1.8 cont
'''
FrequentWordsWithMismatches(Text, k, d)
    Patterns ← an array of strings of length 0
    freqMap ← empty map
    n ← |Text|
    for i ← 0 to n - k
        Pattern ← Text(i, k)
        neighborhood ← Neighbors(Pattern, d)
        for j ← 0 to |neighborhood| - 1
            neighbor ← neighborhood[j]
            if freqMap[neighbor] doesn't exist
                freqMap[neighbor] ← 1
            else
                freqMap[neighbor] ← freqMap[neighbor] + 1
    m ← MaxMap(freqMap)
    for every key Pattern in freqMap
        if freqMap[Pattern] = m
            append Pattern to Patterns
    return Patterns
'''
def frequent_words_with_mismatches(text: str, k: int, d: int) -> list[str]:
    patterns = []
    freqMap = {}
    n = len(text)
    for i in range(0, n - k+1):
        pattern = text[i:i+k]
        neighborhood = neighbors(pattern, d)
        #print((neighborhood))
        for j in range(0,len(neighborhood)):
            neighbor = neighborhood[j]
            if neighbor not in freqMap:
                freqMap[neighbor] = 1
            else:
                freqMap[neighbor] += 1
    m = max(freqMap.values())
    for pattern in freqMap:
        if freqMap[pattern] == m:
            patterns.append(pattern)
    return patterns
#print(frequent_words_with_mismatches("ACGTTGCATGTCGCATGATGCATGAGAGCT",4,1)) # ATGT GATG ATGC
#print(frequent_words_with_mismatches("AGGGT",2,0)) # GG
#print(frequent_words_with_mismatches("AGGCGG",3, 0)) # AGG GGC GCG CGG

# Frequent Words with Mismatches and Reverse Complements Problem
def frequent_words_mismatches_reverse_complements(text: str, k: int, d: int) -> list[str]:
    pass    
