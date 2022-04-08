from sympy import *


''' We show: if m is magic, then m is determined from (d, g, i) '''


a, b, c, d, e, f, g, h, i, s = symbols('a, b, c, d, e, f, g, h, i, s')


m = Matrix([[a, b, c], 
            [d, e, f], 
            [g, h, i]])


eq = [
    s - sum(m.row(0)),
    s - sum(m.row(1)),
    s - sum(m.row(2)),
    s - sum(m.col(0)),
    s - sum(m.col(1)),
    s - sum(m.col(2)),
    s - trace(m),
    s - c - e - g
]


sol = next(iter(linsolve(eq, (a, b, c, e, f, h, s)))) 

print('\n\nFormulas for 3x3 square:\n\n')
for i in range(0, len(sol)):
    print("{} = {}".format(['a', 'b', 'c', 'e', 'f', 'h', 's'][i], sol[i]))
'''
Conclusion:
a = 2*d + 2*g - 3*i
b = -d + 2*i
c = 2*d + g - 2*i
e = d + g - i
f = d + 2*g - 2*i
h = 3*d + 2*g - 4*i
s = 3*d + 3*g - 3*i

We will only use the formula for h
'''


''' We show, using the above formula for h: if each square in the grid is magic, the whole square is determined by 4 numbers 
cross = [[  ,  , b,  ],
         [ c, w, y,  ],
         [  , z, x, a],
         [  , d,  ,  ]]
'''


(aa, bb, cc, dd) = symbols('aa, bb, cc, dd')
(xx, yy, ww, zz) = symbols('xx, yy, ww, zz')

equations = [
    4*bb + yy - 2*xx - 3*aa,
    4*cc + ww - 2*yy - 3*bb,
    4*dd + zz - 2*ww - 3*cc,
    4*aa + xx - 2*zz - 3*dd
]

abcd = next(iter(linsolve(equations, (aa, bb, cc, dd))))

print('\n\nFormulas for other 4 entries in cross, given 4 entries:\n\n')
for i in range(0,4):
    print("{} = {}".format(('aa', 'bb', 'cc', 'dd')[i], abcd[i]))
'''
Conclusion:
aa = 12*ww/35 - 2*xx/35 + 9*yy/35 + 16*zz/35
bb = 9*ww/35 + 16*xx/35 - 2*yy/35 + 12*zz/35
cc = -2*ww/35 + 12*xx/35 + 16*yy/35 + 9*zz/35
dd = 16*ww/35 + 9*xx/35 + 12*yy/35 - 2*zz/35
'''