from operator import mul
from functools import reduce
from itertools import product

comps = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

def prob(competitors, seed):
    if seed not in competitors:
        return 0
    if competitors == [seed]:
        return 1
    
    pairs = list(zip(*[iter(competitors)]*2))
    n = len(competitors)
    g = n // 2

    def new(i):
        digits = [int(d) for d in format(i, f'0{g}b')]
        return [pairs[j][digits[j]] for j in range(g)]

    def prob_new(i):
        probs = [1 - new(i)[j] / sum(pairs[j]) for j in range(g)]
        return reduce(mul, probs, 1)

    return sum([prob_new(i) * prob(new(i), seed) for i in range(2**g)])

p2 = prob(comps, 2)
print(f'p2 = {p2*100:6f}%')

def swap(competitors, i, j):
    aux = list(competitors)
    aux[i], aux[j] = competitors[j], competitors[i]
    return aux

rn = range(16)
ps = [(prob(swap(comps, i, j), 2), i, j) for i, j in product(rn, rn)]
p, i, j = max(ps)
s1, s2 = comps[i], comps[j]
pi = p - p2
print(f'swap seeds: {s1}, {s2}')
print(f'p_swp = {p*100:6f}%')
print(f'p_inc = {pi*100:6f}%')