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
#print(output)

# 10.11 Learning the Parameters of an HMM
import numpy as np

def hmm_parameter_estimation(x, sigma, pi, states):
    sigma_to_idx = {symbol: idx for idx, symbol in enumerate(sigma)}
    state_to_idx = {state: idx for idx, state in enumerate(states)}

    num_states = len(states)
    num_symbols = len(sigma)
    
    transition_matrix = np.zeros((num_states, num_states))
    emission_matrix = np.zeros((num_states, num_symbols))

    # transition    
    for i in range(1, len(pi)):
        prev_state_idx = state_to_idx[pi[i - 1]]
        curr_state_idx = state_to_idx[pi[i]]
        transition_matrix[prev_state_idx][curr_state_idx] += 1
    
    # emission 
    for i in range(len(x)):
        state_idx = state_to_idx[pi[i]]
        symbol_idx = sigma_to_idx[x[i]]
        emission_matrix[state_idx][symbol_idx] += 1
    
    # normalize transition
    for i in range(num_states):
        row_sum = np.sum(transition_matrix[i])
        if row_sum > 0:
            transition_matrix[i] /= row_sum
        else:
            # if no outgoing transitions, distribute evenly
            transition_matrix[i] = 1.0 / num_states
    
    # normalize emission 
    for i in range(num_states):
        row_sum = np.sum(emission_matrix[i])
        if row_sum > 0:
            emission_matrix[i] /= row_sum
        else:
            #if no emissions, distribute evenly
            emission_matrix[i] = 1.0 / num_symbols

    return transition_matrix, emission_matrix

x = "zyzzyzyyyyzzxyyyxzyzzyzxzxzzyzyxzyxzxzyyyyyyzzyzzxxyzzxxzxxxxzzxxxyxzxyxyxzxyyyzxyzzxzzzyyyxyxxyxxyy"
sigma = ['x', 'y', 'z']
pi = "BBAABBBAABABBABAAAABBBBABABBBAAABBAABAAABBAAAAABBAAABAAABBBAABABBBBBABABBAABBAAAAABAABBABAABBBABBBBB"
states = ['A', 'B']

transition_matrix, emission_matrix = hmm_parameter_estimation(x, sigma, pi, states)

'''
print("\t", "\t".join(states))
for i, row in enumerate(transition_matrix):
    print(states[i], "\t", "\t".join(f"{val:.3f}" for val in row))
print("--------")
print("\t", "\t".join(sigma))
for i, row in enumerate(emission_matrix):
    print(states[i], "\t", "\t".join(f"{val:.3f}" for val in row))
'''

# 10.12. Soft Decisions in Parameter Estimation: Q5
import numpy as np

def forward_algorithm(x, sigma, states, transition_matrix, emission_matrix):
    num_states = len(states)
    num_symbols = len(sigma)
    T = len(x)
    
    # init fwd
    forward = np.zeros((T, num_states))
    
    # init
    for k in range(num_states):
        forward[0, k] = emission_matrix[k, sigma.index(x[0])]
    
    # recursive step
    for i in range(1, T):
        for k in range(num_states):
            forward[i, k] = sum(forward[i-1, j] * transition_matrix[j, k] for j in range(num_states)) * emission_matrix[k, sigma.index(x[i])]
    
    return forward

def backward_algorithm(x, sigma, states, transition_matrix, emission_matrix):
    num_states = len(states)
    num_symbols = len(sigma)
    T = len(x)
    
    # init bwd
    backward = np.zeros((T, num_states))
    
    # inti 
    backward[T-1, :] = 1
    
    # recursive step
    for i in range(T-2, -1, -1):
        for k in range(num_states):
            backward[i, k] = sum(transition_matrix[k, j] * emission_matrix[j, sigma.index(x[i+1])] * backward[i+1, j] for j in range(num_states))
    
    return backward

def soft_decoding(x, sigma, states, transition_matrix, emission_matrix):
    num_states = len(states)
    T = len(x)
    
    # fwd
    forward_probs = forward_algorithm(x, sigma, states, transition_matrix, emission_matrix)
    
    # bwd
    backward_probs = backward_algorithm(x, sigma, states, transition_matrix, emission_matrix)
    
    # soft decoding Pr
    soft_decoding_probs = np.zeros((T, num_states))
    
    for i in range(T):
        denominator = sum(forward_probs[i, k] * backward_probs[i, k] for k in range(num_states))
        for k in range(num_states):
            soft_decoding_probs[i, k] = (forward_probs[i, k] * backward_probs[i, k]) / denominator
    
    return soft_decoding_probs

x = "zzyzxxxxxz"
sigma = ['x', 'y', 'z']
states = ['A', 'B', 'C', 'D']
transition_matrix = np.array([[0.332,0.176,0.237,0.255], [0.157,0.325,0.117,0.401], [0.061,0.061,0.574,0.304],[0.566,0.09,0.188,0.156] ])
emission_matrix = np.array([[0.48,0.489,0.031], [0.366,0.433,0.201], [0.091,0.365,0.544], [0.343,0.41,0.247]])

soft_decoding_probs = soft_decoding(x, sigma, states, transition_matrix, emission_matrix)

'''
print("\t", "\t".join(states))
for i in range(len(x)):
    print(f"{soft_decoding_probs[i, 0]:.4f}\t{soft_decoding_probs[i, 1]:.4f}")
'''

# 10.13. Baum-Welch Learning

def forward_algorithm(x, sigma, states, transition_matrix, emission_matrix):
    num_states = len(states)
    T = len(x)
    
    forward = np.zeros((T, num_states))
    
    for k in range(num_states):
        forward[0, k] = emission_matrix[k, sigma.index(x[0])]
    
    for i in range(1, T):
        for k in range(num_states):
            forward[i, k] = sum(forward[i-1, j] * transition_matrix[j, k] for j in range(num_states)) * emission_matrix[k, sigma.index(x[i])]
    
    return forward

def backward_algorithm(x, sigma, states, transition_matrix, emission_matrix):
    num_states = len(states)
    T = len(x)
    
    backward = np.zeros((T, num_states))
    backward[T-1, :] = 1  # init
    
    for i in range(T-2, -1, -1):
        for k in range(num_states):
            backward[i, k] = sum(transition_matrix[k, j] * emission_matrix[j, sigma.index(x[i+1])] * backward[i+1, j] for j in range(num_states))
    
    return backward

def baum_welch(x, sigma, states, transition_matrix, emission_matrix, num_iterations):
    num_states = len(states)
    T = len(x)
    
    for iteration in range(num_iterations):
        forward_probs = forward_algorithm(x, sigma, states, transition_matrix, emission_matrix)
        backward_probs = backward_algorithm(x, sigma, states, transition_matrix, emission_matrix)

        # gamma and xi
        xi = np.zeros((T - 1, num_states, num_states))
        gamma = np.zeros((T, num_states))

        for i in range(T - 1):
            denom = sum(forward_probs[i, a] * transition_matrix[a, b] * emission_matrix[b, sigma.index(x[i+1])] * backward_probs[i+1, b]
                        for a in range(num_states) for b in range(num_states))

            for a in range(num_states):
                gamma[i, a] = sum(forward_probs[i, a] * transition_matrix[a, b] * emission_matrix[b, sigma.index(x[i+1])] * backward_probs[i+1, b]
                                  for b in range(num_states)) / denom

                for b in range(num_states):
                    xi[i, a, b] = (forward_probs[i, a] * transition_matrix[a, b] * emission_matrix[b, sigma.index(x[i+1])] * backward_probs[i+1, b]) / denom

        gamma[T - 1, :] = forward_probs[T - 1, :] * backward_probs[T - 1, :] / np.sum(forward_probs[T - 1, :] * backward_probs[T - 1, :])

        # transition 
        for a in range(num_states):
            for b in range(num_states):
                transition_matrix[a, b] = np.sum(xi[:, a, b]) / np.sum(gamma[:T-1, a])

        # emission 
        for a in range(num_states):
            for symbol in range(len(sigma)):
                emission_matrix[a, symbol] = np.sum(gamma[:, a] * (np.array([s == sigma[symbol] for s in x], dtype=int))) / np.sum(gamma[:, a])

    return transition_matrix, emission_matrix

num_iterations = 100
x = "xzxxzzzzzzzyzxyxzyzzxzyzxxzxzyyzzyxyzxyyyxyyyyxxxzxxxyxzxxxyyzxzzxzzxxzxzzzyxyyzyyyxzxyyyyyxyxxxzzzy"
sigma = ['x', 'y', 'z']
states = ['A', 'B', 'C', 'D']
transition_matrix = np.array([[0.192,0.156,0.339,0.314], [0.374,0.083,0.471,0.072],[0.241,0.415,0.341,0.003], [0.156,0.186,0.326,0.331]])
emission_matrix = np.array([[0.079,0.673,0.248], [0.279,0.129,0.591], [0.445,0.248,0.306],[0.378,0.419,0.203]])

transition_matrix, emission_matrix = baum_welch(x, sigma, states, transition_matrix, emission_matrix, num_iterations)

print("\t", "\t".join(states))
for i, row in enumerate(transition_matrix):
    print(states[i], "\t", "\t".join(f"{val:.3f}" for val in row))
print("--------")
print("\t", "\t".join(sigma))
for i, row in enumerate(emission_matrix):
    print(states[i], "\t", "\t".join(f"{val:.3f}" for val in row))
