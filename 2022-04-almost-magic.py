from ortools.sat.python import cp_model

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
solver.parameters.max_time_in_seconds = 15
solver.Solve(model)

# print the solution
print('solution = {}'.format([solver.Value(i) for i in x.values()]))
print('sum = {}'.format(sum([solver.Value(i) for i in x.values()])))

# solution found:
# solution = [14, 17, 3, 1, 11, 22, 40, 8, 10, 19, 5, 9, 23, 37, 6, 12,
#             16, 39, 7, 24, 18, 4, 13, 21, 29, 34, 2, 26]
# sum = 470
