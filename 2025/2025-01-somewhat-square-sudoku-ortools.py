import numpy as np
from codetiming import Timer
from ortools.sat.python import cp_model
from sympy import divisors

# fmt: off
z = -1
input_grid = np.array([
    [z, z, z, z, z, z, z, 2, z],
    [z, z, z, z, 2, z, z, z, 5],  # The 2 in this row is implied by the other 2s
    [z, 2, z, z, z, z, z, z, z],
    [z, z, 0, z, z, z, z, z, z],
    [z, z, z, z, z, z, z, z, z],
    [z, z, z, 2, z, z, z, z, z],
    [z, z, z, z, 0, z, z, z, z],
    [z, z, z, z, z, 2, z, z, z],
    [z, z, z, z, z, z, 5, z, z],
], dtype=np.int32)
# fmt: on

known = list(np.unique(input_grid[input_grid >= 0]))
omitable = [d for d in range(10) if d not in known]
gcd_lower_bound = 1
gcd_upper_bound = 98765432
num_upper_bound = 987654320
P = 10 ** np.arange(8, -1, -1, dtype=np.int32)


# Solver, variables, objective, constraints
# ----------------------------------------------------------------------
model = cp_model.CpModel()
X = np.array([[model.new_int_var(0, 9, f"x{[i, j]}") for j in range(9)] for i in range(9)])  # grid nums
N = np.array([model.new_int_var(0, num_upper_bound, f"n{[i]}") for i in range(9)])  # row nums
o = model.new_int_var(0, 9, "o")  # number to omit
g = model.new_int_var(gcd_lower_bound, gcd_upper_bound, "g")  # gcd

# Objective
model.maximize(g)

# Must omit one of the 9 digits
for x in X.flat:
    model.add(x != o)

# Given values
for x, y in zip(X.flat, input_grid.flat):
    if y >= 0:
        model.add(x == y)

# g divides each individual row => g divides sum of rows
# sum of rows = 111111111 * (sum of digits) = 111111111 * (1 + ... + 9 - o) = 111111111 * (45 - o)
# it is not necessary to add this constraint, but it makes the solver go significantly faster
assignments = [(o_, g_) for o_ in range(10) for g_ in divisors(111111111 * (45 - o_))]
model.add_allowed_assignments([o, g], assignments)

# Sudoku
corners = list(range(0, 9, 3))
squares = np.vstack([X[i : i + 3, j : j + 3].reshape(-1) for i in corners for j in corners])
stacked = np.vstack([X, X.T, squares])
for xs in stacked:
    model.add_all_different(*xs)

# Divisible
for n, xs in zip(N, X):
    model.add(n == xs @ P)
    model.add(g <= n)
    model.add_modulo_equality(0, n, g)

# Solve problem
# ----------------------------------------------------------------------
status_names: dict[int, str] = {
    cp_model.UNKNOWN: "UNKNOWN",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.OPTIMAL: "OPTIMAL",
}

print(model.validate())
with Timer(initial_text="Solving model..."):
    solver = cp_model.CpSolver()
    status: int = solver.solve(model)
print(f"Status = {status_names[status]}")

if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    xm = np.array([[solver.value(X[i, j]) for j in range(9)] for i in range(9)], dtype=np.int32)
    ns = np.array([solver.value(N[i]) for i in range(9)])
    print(f"Maximum of objective function: {solver.objective_value}")
    print("X = \n", xm, sep="")
    print("N = ", ns)
else:
    print("No solution found.")
# Elapsed time: 0.6110 seconds
# Status = OPTIMAL
# Maximum of objective function: 12345679.0
# X =
# [[3 9 5 0 6 1 7 2 8]
#  [0 6 1 7 2 8 3 9 5]
#  [7 2 8 3 9 5 0 6 1]
#  [9 5 0 6 1 7 2 8 3]
#  [2 8 3 9 5 0 6 1 7]
#  [6 1 7 2 8 3 9 5 0]
#  [8 3 9 5 0 6 1 7 2]
#  [5 0 6 1 7 2 8 3 9]
#  [1 7 2 8 3 9 5 0 6]]
