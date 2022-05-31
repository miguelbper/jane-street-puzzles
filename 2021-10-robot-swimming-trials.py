from math import factorial
from scipy.special import comb

# stirling(n, k) = number of partitions of {1,...,n} into k sets
def stirling(n, k):
    return sum((-1)**i * comb(k, i) * (k - i)**n  for i in range(k+1)) / factorial(k)

# probability that a robot playing uniform wins, if everyone else is playing discrete
# = 1 - probability that it loses
# = 1 - probability that every race is filled
# = 1 - (number of ways of placing 3n-1 robots into n races, s.t. every race is filled) / (number of ways of placing 3n-1 robots into n races)
def p(n):
    return 1 - factorial(n) * stirling(3*n-1, n) / n**(3*n-1)

# print values
for n in range(1,100):
    print('n = {}, p = {:.6f}'.format(n, p(n)))
    if p(n) > 1/3:
        break