from math import comb
from collections import Counter

'''
Facts: 
    1. The center of each quarter-circle is in Z^2.
    2. The radius of each quarter circle is 1.
    3. The endpoints of every quarter circle are in Z^2.
Explanation: The problem statement does not explicitly say that these 
conditions must hold. But, it does say that there are 36 curves 
enclosing a region with area 4 - π. These curves correspond to the 36 
possible translations of the given curve. If condition 1, 2, or 3 was
removed, we could create other curves enclosing a region of area 4 - π. 


Consider the following grid:

     0    1    2    3    4    5    6    7
  +----+----+----+----o----+----+----+----+
0 |    |    |    |    |    |    |    |    |
  +----+----+----o----+----o----+----+----+
1 |    |    |    |  0 |  1 |    |    |    |
  +----+----o----+----+----+----o----+----+
2 |    |    |  2 |  3 |  4 |  5 |    |    |
  +----o----+----+----+----+----+----o----+
3 |    |  6 |  7 |  8 |  9 | 10 | 11 |    |
  +----o----+----+----+----+----+----o----+
4 |    |    | 12 | 13 | 14 | 15 |    |    |
  +----+----o----+----+----+----o----+----+
5 |    |    |    | 16 | 17 |    |    |    |
  +----+----+----o----+----o----+----+----+
6 |    |    |    |    |    |    |    |    |
  +----+----+----+----o----+----+----+----+

In this grid, the 'o's represent the boundary of the given 7x7 grid. 
Every quarter-circle has endpoints in a 'o' or '+' of the grid above.
The quarter-circle is determined by its endpoints, plus one bit of data 
encoding whether the quarter-circle is convex or concave. 


For a curve C as in the problem statement, denote:
    S(C) = curve obtained from C by replacing every quarter-circle by 
           a line segment with the same endpoints
    [s_0,...,s_{n-1}] = list where 
                        n = number of segments in S(C) 
                        s_i = +1 if the ith quarter-circle is convex
                        s_i = -1 if the ith quarter-circle is concave

Fact: 32 = Area(C) = Area(S(C)) and s_0 + ... + s_{n-1} = 0.
Proof: Denote by Q a quarter circle with radius 1 and by T a triangle 
with side 1. 

    32 = Area(C)
       = Area(S(C)) + Σ_{i=0}^{n-1} s_i (Area(Q) - Area(T))
       = Area(S(C)) + (π - 2)/4 Σ_{i=0}^{n-1} s_i

If Σ_{i=0}^{n-1} s_i != 0, the above computation would show that π is a 
rational number, which is not true. Therefore, Σ_{i=0}^{n-1} s_i = 0 and
32 = Area(C) = Area(S(C)).


So, choosing C as in the problem statement is equivalent to choosing 
S(C) enclosing an area of 32, along with signs [s_0,...,s_{n-1}] such 
that Σ_{i=0}^{n-1} s_i = 0.

The squares {0,...,17} have an area of 36, so choosing an area of 32 
is equivalent to choosing 2 squares k, l from {0,...,17} to be removed. 
See the function grid below.

The curve C is simple, which means that it splits the grid into two 
connected components. See the function numcc below.

For each choice of area of 32 (with two connected components), we need
to count the number of lists [s_0,...,s_{n-1}] such that 
Σ_{i=0}^{n-1} s_i = 0 (where n = perimeter of the region). Choosing such 
a list is equivalent to choosing half of the s_i to be +1 (and the rest
is -1). In other words, this count is comb(n, n // 2). If n is odd then
the count should be 0. See the functions perimeter and count below.


Combining every statement above, we conclude that
sol = 2 Σ_{0<=k<l<18, numcc(grid(k,l)) = 2} count(perimeter(grid(k,l)))

Here, we need to multiply the sum by 2 because there are 2 possible ways
we could align the 6x7 grid we drew above with the original 7x7 grid.
'''


def coordinates(k: int) -> tuple[int, int]:
    ''' Given k in {0,...,17}, return coordinates (i, j) of the square 
    labeled k (in the grid drawn above). '''
    return divmod(k + 11 + 5*(k > 1) + 3*(k > 5) + 3*(k > 11) + 5*(k > 15), 8)


def grid(k: int, l: int) -> list[list[int]]:
    ''' Given k, l in {0,...,17}, return a 7x8 array xss where
        xss[i][j] = 1 if (i, j) has label m in {0,...,17} - {k, l}
                    0 otherwise. '''
    xss = [[0 for _ in range(8)] for _ in range(7)]
    for m in range(18):
        if m != l and m != k:
            mi, mj = coordinates(m)
            xss[mi][mj] = 1
    return xss


def numcc(xss: list[list[int]]) -> int:
    ''' Given a 7x8 array xss, return the num of connected components
    (this is computed using a depth first search). Connectedness is 
    defined as follows: if vertices (i0, j0) and (i1, j1) are such that
        1. distance((i0, j0), (i1, j1)) = 1
        2. xss[i0][j0] = xss[i1][j1]
    then (i0, j0) and (i1, j1) are in the same connected component.
    '''
    direc = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    verts = {(i, j) for i in range(7) for j in range(8)}
    ans = 0
    while verts:
        ans += 1
        v = verts.pop()
        stack = [v]
        while stack:
            i, j = stack.pop()
            for di, dj in direc:
                x = i + di
                y = j + dj
                if (x, y) in verts and xss[i][j] == xss[x][y]:
                    verts.remove((x, y))
                    stack.append((x, y))
    return ans


def perimeter(xss: list[list[int]]) -> int:
    ''' Given a 7x8 array xss (with 2 connected components), return
    the perimeter of the area with xss[i][j] = 1. '''
    ans = 0
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for k in range(18):
        i, j = coordinates(k)
        n_sides = sum(xss[i + di][j + dj] == 0 for di, dj in dirs)
        ans += (xss[i][j] == 1) * n_sides
    return ans


def count(p: int) -> int:
    ''' Given a perimeter p, return the num of lists [s_0,...,s_{p-1}]
    such that s_0 + ... + s_{p-1} = 0. 
    
    Remark: the possible values of the perimeter are {18, 20, 22, 24},
    so this function could be just return comb(p, p // 2).
    '''
    q, r = divmod(p, 2)
    return 0 if r else comb(p, q)


# Use the functions above to compute the solution
grids = (grid(k, l) for l in range(18) for k in range(l))
counter = Counter(perimeter(xss) for xss in grids if numcc(xss) == 2)
solution = 2 * sum(count(p) * n for p, n in counter.items())

print(f'{solution = }')
# solution = 89519144