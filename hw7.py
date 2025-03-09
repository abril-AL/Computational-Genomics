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
#print(result)

# Virterbi
import numpy as np

def viterbi_algorithm(x, alphabet, states, transition, emission):
    # Convert states and alphabet to indices
    state_index = {state: i for i, state in enumerate(states)}
    alphabet_index = {char: i for i, char in enumerate(alphabet)}
    
    # Number of states and observations
    num_states = len(states)
    num_obs = len(x)
    
    # Initialize the Viterbi table (log probabilities for numerical stability)
    dp = np.zeros((num_states, num_obs))
    backpointer = np.zeros((num_states, num_obs), dtype=int)

    # Initialize the first column (initial state probabilities)
    for state in range(num_states):
        dp[state, 0] = np.log(emission[state, alphabet_index[x[0]]])  # Log of emission probability for the first observation
        # For the first observation, we assume equal transition probabilities from the start
        dp[state, 0] += np.log(1 / num_states)  # Log of transition probability (uniform from initial state)
    
    # Fill the Viterbi table
    for t in range(1, num_obs):
        for state in range(num_states):
            # Calculate the maximum probability for each state at time t
            max_prob = -np.inf
            max_state = -1
            for prev_state in range(num_states):
                prob = dp[prev_state, t-1] + np.log(transition[prev_state, state]) + np.log(emission[state, alphabet_index[x[t]]])
                if prob > max_prob:
                    max_prob = prob
                    max_state = prev_state
            dp[state, t] = max_prob
            backpointer[state, t] = max_state
    
    # Backtrack to find the best path
    best_path = [0] * num_obs
    best_state = np.argmax(dp[:, -1])
    best_path[-1] = best_state
    
    for t in range(num_obs - 2, -1, -1):
        best_state = backpointer[best_state, t+1]
        best_path[t] = best_state
    
    # Convert the state indices back to the state labels
    best_path_labels = ''.join([states[state] for state in best_path])
    return best_path_labels


# Example usage with your provided input
x = "yyzzzzyxyyzzxzzyzzyzxxxxxyxyxzxzyxzxyyzyyzxyxzyxzzxyzxzzzzzyzzzzzzzyyyyyxxxxxxxxyyzxxxxyxyxzxzzyyyxy"
alphabet = ['x', 'y', 'z']
states = ['A', 'B','C','D']
transition = np.array([
    [0.35,0.045,0.149,0.456],
    [0.538,0.305,0.049,0.108],
    [0.504,0.4,0.07,0.026],
    [0.293,0.031,0.585,0.091]
])
emission = np.array([
    [0.131,0.759,0.11],
    [0.585,0.141,0.274],
    [0.509,0.103,0.388],
    [0.358,0.241,0.401]
])

output = viterbi_algorithm(x, alphabet, states, transition, emission)
#print(output)

# 10.7 Solve the Outcome Likelihood Problem.
import numpy as np
def outcome_likelihood(x, alphabet, states, transition, emission):
    state_index = {state: i for i, state in enumerate(states)}
    alphabet_index = {char: i for i, char in enumerate(alphabet)}
    
    num_states = len(states)
    num_obs = len(x)
    
    # init fwd table
    forward = np.zeros((num_states, num_obs))
    
    # first observation
    for state in range(num_states):
        forward[state, 0] = (1 / num_states) * emission[state, alphabet_index[x[0]]]
    
    # fill in the forward table 
    for i in range(1, num_obs):
        for k in range(num_states):
            forward[k, i] = sum(forward[l, i-1] * transition[l, k] * emission[k, alphabet_index[x[i]]] for l in range(num_states))
    
    # sum the probabilities for all states 
    probability = np.sum(forward[:, -1])
    
    return probability

x = "zyyzzyyxxzxyzzxzyxyzyzxxzzyxyyzxxxyyzxzzzxxyzxyyxxzyxyxxyyxzzyzzyyxyzyyzzyyzzyzyyxyyyzxxxyyxzxyxzyyx"
alphabet = ['x', 'y', 'z']
states = ['A', 'B', 'C']
transition = np.array([
    [0.203,0.665,0.132],  
    [0.461,0.406,0.133],  
    [0.133,0.515,0.352]
])
emission = np.array([
    [0.135,0.448,0.417],  
    [0.542,0.205,0.253],  
    [0.62,0.226,0.154]
])

output = outcome_likelihood(x, alphabet, states, transition, emission)
print(output)
