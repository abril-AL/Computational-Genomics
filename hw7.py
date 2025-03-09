# Homework 7
# 10.5. Hidden Markov Models: Q8* (recommended), Q10 
# 10.6. The Decoding Problem: Q7* (recommended) 
# 10.7. Finding the Most Likely Outcome of an HMM: Q4* (recommended) 
# 10.11. Learning the Parameters of an HMM: Q4 
# 10.12. Soft Decisions in Parameter Estimation: Q5 
# 10.13. Baum-Welch Learning: Q5

# Probability of an Outcome Given a Hidden Path Problem
def compute_emission_probability(emitted_sequence, hidden_path, emission_matrix):
    probability = 1.0
    for obs, state in zip(emitted_sequence, hidden_path):
        probability *= emission_matrix[state][obs]  # Multiply probabilities along the path
    return probability

emitted_sequence = "yyzzxxzzxxzyyzxyzyxyyyxyxzyxxyxxyyzxyyzxxzyyzzyzzy  "
hidden_path = "BAABAAABAABBBBBABBAAABBBABBBBABBAAAABAABBBBBBABAAA"

emission_matrix = {
    "A": {"x": 0.099, "y": 0.58, "z": 0.321},
    "B": {"x": 0.098, "y": 0.654, "z": 0.248}
}

result = compute_emission_probability(emitted_sequence, hidden_path, emission_matrix)
print(result)
