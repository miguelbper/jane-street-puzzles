from functools import cache

# 1. footbal
# ----------------------------------------------------------------------
'''
Solution: expected number of steps = 20
Proof: Imagine a random walk of the ant, which continues forever. There
are 20 hexagons. By symmetry, 1/20 of the time the ant is in its
starting hexagon. The average time between two consecutive occurences of
the ant being in the starting hexagon is 20.
'''


# 2. kitchen
# ----------------------------------------------------------------------
'''
Consider the following coordinates (i, j)

       (-1, 1)   (-1, 0)        B         (-1, -1)
(0, 1)         B         (0, 0)   (0, -1)
       ( 1, 1)   ( 1, 0)        B         ( 1, -1)
(2, 1)         B         (2, 0)   (2, -1)
'''


@cache
def prob(i, j, n):
    '''
    probability that the ant can reach (0, 0) starting at (i ,j)
    in n steps (remaining, out of 20 total steps)
    '''
    if (i, j) == (0, 0) and n < 20:
        return 1
    if (i, j) != (0, 0) and n == 0:
        return 0

    p1 = prob(i - 1, j, n - 1)              # left
    p2 = prob(i + 1, j, n - 1)              # right
    p3 = prob(i, j - (-1)**(i + j), n - 1)  # up / bottom

    return (p1 + p2 + p3) / 3


print(f'p = {1 - prob(0, 0, 20):.7f}')
# p = 0.4480326
