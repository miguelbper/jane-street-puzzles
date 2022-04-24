import numpy as np
from itertools import permutations
from functools import partial
from tqdm import tqdm

# by formulas.py, the u-square is determined as follows
def au(u, l, d, r):
    return -l + d

def bu(u, l, d, r):
    return 2*u - 2*l + d - r

def sq(u, l, d, r):
    a = au(u, l, d, r)
    b = bu(u, l, d, r)
    c = u
    return np.array([[c     - b, c + a + b, c - a    ],
                     [c - a + b, c        , c + a - b],
                     [c + a    , c - a - b, c     + b]])

def rotate(n, m):
    for i in range(n % 4):
        m = np.array([np.transpose(m[:,2]), 
                      np.transpose(m[:,1]), 
                      np.transpose(m[:,0])])
    return m

# therefore the grid is determined as follows 
# (where we are using rotational symmetry)
def grid(u, l, d, r):
    m = np.zeros([6,6], dtype = "int")

    m[0:3,1:4] = rotate(0, sq(u, l, d, r))
    m[2:5,0:3] = rotate(1, sq(l, d, r, u))
    m[3:6,2:5] = rotate(2, sq(d, r, u, l))
    m[1:4,3:6] = rotate(3, sq(r, u, l, d))

    return m

def no_dups(g):
    g = g[g!=0]
    return 0 == len(g) - len(np.unique(g))

# it can be shown that the u-square has entries which are all positive iff
# (we can use rotational symmetry to call this function for the other squares as well)
def positive(u, l, d, r):
    return u > abs(au(u, l, d, r)) + abs(bu(u, l, d, r))

# it can be shown that if s = 7 * (u + l + d + r), 
# where s is the sum of all the entries
# let k = s / 7 = u + l + d + r
def valid(k, tup):
    (u, l, d) = tup
    r = k - u - l - d
    if r <= 0:
        return False
    if r in [u, l, d]:
        return False
    if u != min(u, l, d, r):
        return False
    if not positive(u, l, d, r):
        return False
    if not positive(l, d, r, u):
        return False
    if not positive(d, r, u, l):
        return False
    if not positive(r, u, l, d):
        return False
    return True

# if (u, l, d, r) are such that u + l + d + r = k, 
# then it is possible to show that u, l, d, r <= min(k - 6, 7*(k - 54))
def iterator(k):
    m = min(k - 6, 7*(k - 54))
    val = partial(valid, k)
    itr = permutations(range(1, m + 1), 3)
    itr = filter(val, itr)
    return itr

# find the first grid such that u + l + d + r = k (if it exists)
def find_sum(k):
    for (u, l, d) in iterator(k):
        r = k - u - l - d
        g = grid(u, l, d, r)
        if no_dups(g):
            return g

# loop over k to find the magic grid with the lowest sum
# s_0 = 1 + 2 + ... + 28 = 406. k_0 = 406 / 7 = 58
def magic(n):
    for k in tqdm(range(58, 58 + n)):
        g = find_sum(k)
        if isinstance(g, np.ndarray):
            return g

x = magic(100)

print(x)