from scipy.optimize import fsolve

e = lambda p: 1 - p - (1 - p/8)**24
p = fsolve(e, 1)

print(p)
