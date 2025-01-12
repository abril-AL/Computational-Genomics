###############
# CS 122 HW 1 #
###############

import sys


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


