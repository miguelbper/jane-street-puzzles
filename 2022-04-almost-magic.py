from ortools.sat.python import cp_model

# ======================================================================
# Solution 1: iterate to find numbers making all squares magic
# ======================================================================
print('Solution 1: lowest sum with all squares magic')

'''
Consider the following square:
o p q
r s n
l t m

If the square is magic, then o, p, q, r, s are given in terms of l, m, n
by the equations in the function cells below. Moreover,
t = -4*g + 2*m + 3*f
'''
def cells(l, m, n):
    o = -2*l +   m + 2*n
    p =  2*l       -   n
    q = -3*l + 2*m + 2*n
    r = -2*l + 2*m +   n
    s = -  l +   m +   n
    return o, p, q, r, s


'''
Consider the full grid
0 ? ? ? 0 0
0 ? ? x ? ?
? y b a ? ?
? ? c d z ?
? ? w ? ? 0
0 0 ? ? ? 0

By the equations above, if we know x, y, w, z, a, b, c, d then we know
the entire square. Apply the equation t = -4*g + 2*m + 3*f from above
to a, b, c, d:
4*y + b - 2*a - 3*x = 0
4*w + c - 2*b - 3*y = 0
4*z + d - 2*c - 3*w = 0
4*x + a - 2*d - 3*z = 0

We can solve this system and obtain x, y, w, z in terms of a, b, c, d.
The solution is the function xywz below.

Using modular arithmetic, it is possible to show that x, y, w, z are
integers iff (d - 22*a - 6*b - 8*c) % 35 == 0.
'''
def xywz(a, b, c, d):
    x = (-2*a + 9*b + 12*c + 16*d) // 35
    y = (-2*b + 9*c + 12*d + 16*a) // 35
    w = (-2*c + 9*d + 12*a + 16*b) // 35
    z = (-2*d + 9*a + 12*b + 16*c) // 35
    return x, y, w, z


# indices of cells in each square which are not x, y, w, z, a, b, c, d
cells_ind = {
    'right': [7, 13, 19, 6, 12],
    'up': [0, 1, 2, 3, 4],
    'down': [20, 14, 8, 21, 15],
    'left': [27, 26, 25, 24, 23],
}


'''
Now we have equations for every cell of the grid in terms of a, b, c, d.
Let s = sum of every cell and k = a + b + c + d.
It is possible to show that s = 7*k.
We will loop through all possible values of k, a, b, c, d in the
generator gen() below. By symmetry, we may assume a = min(a, b, c, d).
'''
def gen():
    # sum = 7 * k, s >= 1 + ... + 28 = 406, s <= 1111
    for k in range(58, 158):
        # a = k - b - c - d <= k - 2 - 3 - 4 = k - 9
        for a in range(1, k - 8):
            # b = k - a - c - d <= k - a - (a+1) - (a+2) = k - 3*(a+1)
            for b in range(a + 1, k - 3*(a + 1) + 1):
                # c = k - a - b - d <= k - a - b - (a+1) = k - 2*a - b-1
                for c in range(a + 1, k - 2*a - b):
                    d = k - a - b - c
                    # x, y, w, z are ints <=> (...) % 35 == 0
                    if (d - 22*a - 6*b - 8*c) % 35 == 0:
                        yield (k, a, b, c, d)


# loop to find the lowest sum with all squares magic
for k, a, b, c, d in gen():
    # compute x, y, w, z from a, b, c, d
    x, y, w, z = xywz(a, b, c, d)
    if any(j <= 0 for j in [x, y, w, z]):
        continue

    # compute other cells
    cells_arg = {
        'right': (x, d, z),
        'up': (y, a, x),
        'down': (w, b, y),
        'left': (z, c, w),
    }
    cells_out = dict()
    for key in ['right', 'up', 'down', 'left']:
        cells_out[key] = cells(*cells_arg[key])
        if any(j <= 0 for j in cells_out[key]):
            break
    else:
        # if no break, proceed here
        # compute list from the all the values
        l = list(range(28))
        l[11], l[10], l[16], l[17] = a, b, c, d
        l[ 5], l[ 9], l[22], l[18] = x, y, w, z
        for key in ['right', 'up', 'down', 'left']:
            for i, j in enumerate(cells_ind[key]):
                l[j] = cells_out[key][i]

        # check that all elements are distinct
        if len(l) == len(set(l)):
            print(f'solution = {l}\nsum = {7 * k}')
            break
# solution = [39, 3, 24, 7, 22, 37, 45, 11, 23, 20, 41, 5, 31, 57, 46,
#             28, 10, 51, 17, 25, 15, 36, 33, 26, 19, 35, 1, 42]
# sum = 749


# ======================================================================
# Solution 2: Use OR-Tools to do constraint optimization
# ======================================================================
print('\nSolution 2: lowest sum with all squares almost magic')
SQUARES = ['u', 'l', 'd', 'r']

sq = {}
sq['u'] = [[ 0,  1,  2], [ 3,  4,  5], [ 9, 10, 11]]
sq['r'] = [[ 5,  6,  7], [11, 12, 13], [17, 18, 19]]
sq['l'] = [[ 8,  9, 10], [14, 15, 16], [20, 21, 22]]
sq['d'] = [[16, 17, 18], [22, 23, 24], [25, 26, 27]]


def eqs(sq):
    cols = [[sq[i][j] for i in range(3)] for j in range(3)]
    diag = [[sq[0][0], sq[1][1], sq[2][2]], [sq[0][2], sq[1][1], sq[2][0]]]
    return sq + cols + diag


# define model
model = cp_model.CpModel()

# define variables
x = {j: model.NewIntVar(1, 60, 'x[%i]' % j) for j in range(28)}
s = {
    k: {
        e: model.NewIntVar(1, 180, 's[{}][{}]'.format(k, e)) for e in range(8)
    } for k in SQUARES
}
ma = {k: model.NewIntVar(1, 180, 'ma[{}]'.format(k)) for k in SQUARES}
mi = {k: model.NewIntVar(1, 180, 'mi[{}]'.format(k)) for k in SQUARES}

# add goal: we want the sum of the entries to be minimal
model.Minimize(cp_model.LinearExpr.Sum(x.values()))

# add constraints
# every number should be different
model.AddAllDifferent(list(x.values()))

# every square k should be almost magic
for k in SQUARES:
    model.Add(ma[k] - mi[k] <= 1)
    model.AddMaxEquality(ma[k], s[k].values())
    model.AddMinEquality(mi[k], s[k].values())
    for i in range(8):
        vars = dict((j, x[j]) for j in eqs(sq[k])[i]).values()
        model.Add(cp_model.LinearExpr.Sum(vars) == s[k][i])

# solve the problem
# Lowest sum is 470. If that sol is not found, try increasing max_time.
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60
solver.Solve(model)

# print the solution
print('solution = {}'.format([solver.Value(i) for i in x.values()]))
print('sum = {}'.format(sum([solver.Value(i) for i in x.values()])))

# solution found:
# solution = [14, 17, 3, 1, 11, 22, 40, 8, 10, 19, 5, 9, 23, 37, 6, 12,
#             16, 39, 7, 24, 18, 4, 13, 21, 29, 34, 2, 26]
# sum = 470
