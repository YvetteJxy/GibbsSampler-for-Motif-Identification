# -*- coding: utf-8 -*-

from numpy.random import randint
from sampling import sample

# Read data from file
filename = 'a.seq'
f = open(filename, 'r')

K = int(f.readline())  # Number of sequences
N = int(f.readline())  # Length of each sequence
w = int(f.readline())  # Length of motif
alphabet = list(f.readline()[:-1])  # Possible characters: ['A', 'T', 'C', 'G']
alpha_b = f.readline()  # Not too important
alpha_w = f.readline()  # Not too important

sequences = []
for i in range(K):  # Reading each sequence
    seq = f.readline()[:-1].split(',')
    sequences += [seq]

position = list(map(int, f.readline()[:-1].split(',')))  # Reading initial motif positions
f.close()


def compute_model(sequences, pos, alphabet, w):
    q = {x: [1] * w for x in alphabet}  # Probability matrix for motif positions
    p = {x: 1 for x in alphabet}  # Background probability

    # Counting character occurrence in motif positions
    for i in range(len(sequences)):
        start_pos = pos[i]
        for j in range(w):
            c = sequences[i][start_pos + j]
            q[c][j] += 1
    # Normalize motif position probabilities
    for c in alphabet:
        for j in range(w):
            q[c][j] = q[c][j] / float(K + len(alphabet))

    # Counting character occurrence in background
    for i in range(len(sequences)):
        for j in range(len(sequences[i])):
            if j < pos[i] or j > pos[i] + w:
                c = sequences[i][j]
                p[c] += 1
    # Normalize background probabilities
    total = sum(p.values())
    for c in alphabet:
        p[c] = p[c] / float(total)

    return q, p


# Initialize the state (motif positions) randomly
pos = [randint(0, N - w + 1) for x in range(K)]
THRESHOLD = 1  # Convergence threshold for position changes
converged = False
MAX_ITER = 100  # Set a large max iteration in case convergence doesn't happen

# Loop until convergence or max iterations
for it in range(MAX_ITER):
    old_pos = pos[:]  # Save current positions for comparison
    for i in range(K):
        # Exclude current sequence from model calculation
        seq_minus = sequences[:];
        del seq_minus[i]
        pos_minus = pos[:];
        del pos_minus[i]
        q, p = compute_model(seq_minus, pos_minus, alphabet, w)

        # Calculate probabilities for each possible position in sequence i
        qx = [1] * (N - w + 1)
        px = [1] * (N - w + 1)
        for j in range(N - w + 1):
            for k in range(w):
                c = sequences[i][j + k]
                qx[j] *= q[c][k]  # Motif probability matrix
                px[j] *= p[c]  # Background probability matrix

        # Compute the ratio between motif and background
        Aj = [x / y for (x, y) in zip(qx, px)]
        norm_c = sum(Aj)
        Aj = list(map(lambda x: x / norm_c, Aj))  # Normalize to get probabilities

        # Sample new position for sequence i
        pos[i] = sample(range(N - w + 1), Aj)

    # Check for convergence
    pos_changes = sum(abs(pos[i] - old_pos[i]) for i in range(K))
    if pos_changes < THRESHOLD:
        converged = True
        break

if converged:
    print(f'Converged after {it + 1} iterations')
else:
    print(f'Did not converge after {MAX_ITER} iterations')

# Final motif positions
print('Final motif positions:', pos)
