import numpy as np
from codetiming import Timer
from ortools.sat.python import cp_model


def square(i: int, j: int) -> np.ndarray:
    """Return a boolean matrix with True in [i:i+3, j:j+3]."""
    sq = np.full((6, 6), False)
    sq[i : i + 3, j : j + 3] = True
    return sq


corners = [(0, 1), (2, 0), (3, 2), (1, 3)]
grid = np.logical_or.reduce([square(i, j) for i, j in corners])


# Solver, variables, objective, constraints
# ----------------------------------------------------------------------
model = cp_model.CpModel()
X = np.array([[model.new_int_var(0, 60, f"x[{i}, {j}]") for j in range(6)] for i in range(6)])

# Objective
model.minimize(sum(X.flat))

# Range for the variables
for x in X[~grid]:
    model.add(x == 0)
for x in X[grid]:
    model.add(x != 0)

# Numbers in grid should be distinct
model.add_all_different(*X[grid])

# Each square is almost magic
for i, j in corners:
    X_sq = X[i : i + 3, j : j + 3]
    rows = list(np.sum(X_sq, axis=1))
    cols = list(np.sum(X_sq, axis=0))
    diag = [np.trace(X_sq), np.trace(np.fliplr(X_sq))]
    sums = np.array(rows + cols + diag).reshape(-1, 1)
    diff = np.triu(sums - sums.T)
    for d in diff.flat:
        model.add(d <= 1)
        model.add(d >= -1)

# Break symmetry, so that solution is unique (=> faster)
model.add(X[3, 3] < X[2, 2])
model.add(X[3, 3] < X[2, 3])
model.add(X[3, 3] < X[3, 2])

# Solve problem
# ----------------------------------------------------------------------
status_names = {
    cp_model.UNKNOWN: "UNKNOWN",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.OPTIMAL: "OPTIMAL",
}

with Timer(initial_text="Solving model..."):
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    status = solver.solve(model)
print(f"Status = {status_names[status]}")

if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    xm = np.array([[solver.value(X[i, j]) for j in range(6)] for i in range(6)], dtype=np.int32)
    print(f"Minimum of objective function: {solver.objective_value}")
    print("X = \n", xm, sep="")
else:
    print("No solution found.")
# Solving model...
# Elapsed time: 3.9020 seconds
# Status = OPTIMAL
# Minimum of objective function: 470.0
# X =
# [[ 0 26  2 34  0  0]
#  [ 0 29 21 13  4 18]
#  [24  7 39 16 12  6]
#  [37 23  9  5 19 10]
#  [ 8 40 22 11  1  0]
#  [ 0  0  3 17 14  0]]
