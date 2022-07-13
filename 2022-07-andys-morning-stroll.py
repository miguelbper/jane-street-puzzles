import numpy as np
from numpy.linalg import matrix_power
from scipy import linalg
from math import ceil

# -------------------------------------------
# 1. Compute expected number of steps in ball
# -------------------------------------------

# The soccer ball can be modeled as the following Markov chain
# There are 6 states {0, 1, 2, 3, 4, 5}, where each number is the distance to the starting hexagon
# We model the problem as if we have already taken the first step and start from 1
# If we ever go back to 0, then we stay there
P = np.array([[3, 0, 0, 0, 0, 0],
              [1, 0, 2, 0, 0, 0],
              [0, 1, 1, 1, 0, 0],
              [0, 0, 1, 1, 1, 0],
              [0, 0, 0, 2, 0, 1],
              [0, 0, 0, 0, 3, 0]]) / 3

# If k_i := E(time to hit 0 starting from i), then
# k_0 = 0
# k_i = 1 + sum_{j != 0} P_{ij} k_j

# Define Q_{ij} = P_{ij} if i != 0 and j != 0, Q_{ij} = 0 otherwise 
# Then the above system of equations is equivalent to
# k = [0 1 ... 1]^T + Q k
# We will solve this for k and compute k[1]. Expected number of steps = k[1] + 1

Q = np.block([[np.zeros((1, 1)), np.zeros((1, 5))],
              [np.zeros((5, 1)), P[1:,1:]]])

a = np.eye(6) - Q
b = np.array([0, 1, 1, 1, 1, 1])
k = linalg.solve(a, b)

s = round(k[1])
print(f's = {s} => expected number of steps = {s + 1}')
# s = 19 => expected number of steps = 20

# ---------------------------------
# 2. Compute probability in kitchen
# ---------------------------------

# As before, model the problem with a state for each distance to the starting hexagon
# Assume also that we have already taken the first step (so we start from state 1 as before)
# H := minimal number of steps needed to reach 0 starting from 1 (this is a random variable)

# p
# = Prob(H + 1 > s + 1)
# = Prob(H > s)
# = 1 - Prob(H <= s)
# = 1 - Prob(start at 1, take s steps, end at 0)  [*]
# = 1 - (l * P**s)[0]

# where in step [*] we used the fact that in our model, if Andy goes to 0 then he stays at 0
# and the last equality is a standard fact about Markov chains

# For the purposes of computing Prob(start at 1, take s steps, end at 0), we may consider that the kitchen is finite 
# and that the maximal distance of a hexagon to 0 is ceil(s / 2) + 1
# This is because if Andy were to go to a distance geq than that, then he would not return to 0 in less than s steps

n = ceil(s / 2) + 1
l = np.concatenate((np.array([0, 1]), np.zeros(n - 1)))

P1 = np.concatenate((np.array([1]), np.zeros(n))).reshape(1, -1)
P4 = np.concatenate((np.zeros(n), np.array([1]))).reshape(1, -1)
P2 = np.concatenate((1/3 * np.eye(n - 1), np.zeros([n - 1, 2])), 1)
P3 = np.concatenate((np.zeros([n - 1, 2]), 2/3 * np.eye(n - 1)), 1)
P  = np.concatenate((P1, P2 + P3, P4))

p = 1 - (l @ matrix_power(P, s))[0]
print(f'p = {p:.7f}')
# p = 0.5083113