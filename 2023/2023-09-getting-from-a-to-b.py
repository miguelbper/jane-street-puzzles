# Imports
# ----------------------------------------------------------------------
from math import log, gcd
from itertools import product


# Hidden sentence
# ----------------------------------------------------------------------
sentence = 'Initially, nobody thought out the huge energy needed. ' \
    'Even quite useful approaches looked suspect, ' \
    'adding tons of thought heretofore even beginning.'

first_letters = ''.join(word[0] for word in sentence.split())
print(first_letters)
# IntothenEqualsatotheb
# n to the n equals a to the b


# Compare with Jenga tower
# ----------------------------------------------------------------------
def solutions(n: int) -> set[tuple[int, int]]:
    '''List of (a, b) s.t. n**n == a**b, with a != 0, b != 0 ints.'''
    if n <= 1:
        return set()

    ans = set()

    # loop over a s.t. exists b s.t. n**n == a**b
    for a in range(2, n**n + 1):
        b = round(log(n**n, a))
        if a**b == n**n:
            ans.add((a, b))
            if b % 2 == 0:
                ans.add((-a, b))

    return ans


rows = [None, float('inf'), 3, 2, 7, 2, 6, 2]

for n, x in list(enumerate(rows))[1:]:
    sols = solutions(n)
    print(f'n = {n} => num_sols = {len(sols)} = {x}, (a, b) in {sols}')

'''
n = 1 => num_sols = 0 = inf, (a, b) in set()
n = 2 => num_sols = 3 = 3, (a, b) in {(-2, 2), (4, 1), (2, 2)}
n = 3 => num_sols = 2 = 2, (a, b) in {(3, 3), (27, 1)}
n = 4 => num_sols = 7 = 7, (a, b) in {(4, 4), (-4, 4), (-16, 2), (-2, 8), (256, 1), (16, 2), (2, 8)}
n = 5 => num_sols = 2 = 2, (a, b) in {(5, 5), (3125, 1)}
n = 6 => num_sols = 6 = 6, (a, b) in {(46656, 1), (36, 3), (-6, 6), (216, 2), (6, 6), (-216, 2)}
n = 7 => num_sols = 2 = 2, (a, b) in {(823543, 1), (7, 7)}
'''


# Compute smallest n with x_n = 2023
# ----------------------------------------------------------------------
'''
Question: What is the smallest n such that there are 2023 ways to write
n**n as a**b?

Definition: f(n) = number of ways to write n**n as a**b

Lemma: For each n, let
    n = p_1**m_1 * p_2**m_2 * ... * p_k**m_k
    n**n = p_1**(n m_1) * p_2**(n m_2) * ... * p_k**(n m_k)
    g = gcd(n m_1, n m_2, ..., n m_k) = n * gcd(m_1, m_2, ..., m_k)
    h = gcd(m_1, m_2, ..., m_k) = p_1**e_1 * p_2**e_2 * ... * p_k**e_k
    g = n * h = p_1**(e_1 + m_1) * ... * p_k**(e_k + m_k)
    u_i = e_i + m_i

    Then,
    f(n) = (2 u_1 + 1) * (u_2 + 1) * ... * (u_k + 1)

Proof: Let S = (n m_1, ..., n m_k)
    f(n) = 2 #(even common divisors of S) + #(odd common divisors of S)
         = 2 #(even divisors of g) + #(odd divisors of g)
         = 2 u_1 (u_2+1) * ... * (u_k+1) + (u_2+1) * ... * (u_k+1)
         = (2 u_1 + 1) * (u_2 + 1) * ... * (u_k + 1)                 QED

We want to find the smallest n such that

    7 * 17 * 17
        = 2023
        = f(n)
        = (2 u_1 + 1) * (u_2 + 1) * ... * (u_k + 1)

The smallest such n is given by
    2 u_1 + 1 = 7  => e_1 + m_1 = u_1 = 3
    u_2 + 1   = 17 => e_2 + m_2 = u_2 = 16
    u_3 + 1   = 17 => e_3 + m_3 = u_3 = 16
    gcd(m_1, m_2, ..., m_k) = p_1**e_1 * p_2**e_2 * ... * p_k**e_k
'''

n_smallest = float('inf')
m_1_, m_2_, m_3_ = 0, 0, 0

for e_1, e_2, e_3 in product(range(4), range(17), range(17)):
    # values of m_1, m_2, m_3 that solve the equations
    m_1 = 3 - e_1
    m_2 = 16 - e_2
    m_3 = 16 - e_3

    # gcd
    h_0 = gcd(m_1, m_2, m_3)
    h_1 = 2**e_1 * 3**e_2 * 5**e_3
    if h_0 != h_1:
        continue

    # n
    n = 2**m_1 * 3**m_2 * 5**m_3

    if n < n_smallest:
        n_smallest = n
        m_1_, m_2_, m_3_ = m_1, m_2, m_3

print(f'\nn = {n_smallest} = 2**{m_1_} * 3**{m_2_} * 5**{m_3_}')
