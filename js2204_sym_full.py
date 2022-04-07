from sympy import *
from pprint import pprint

# Define symbols
(
    m00, m01, m02, m03, m04, m05,
    m10, m11, m12, m13, m14, m15,
    m20, m21, m22, m23, m24, m25,
    m30, m31, m32, m33, m34, m35,
    m40, m41, m42, m43, m44, m45,
    m50, m51, m52, m53, m54, m55
) = symbols('m00, m01, m02, m03, m04, m05, m10, m11, m12, m13, m14, m15, m20, m21, m22, m23, m24, m25, m30, m31, m32, m33, m34, m35, m40, m41, m42, m43, m44, m45, m50, m51, m52, m53, m54, m55')

(a, b, c, d, sr, su, sl, sb) = symbols('a, b, c, d, sr, su, sl, sb')


# Define matrices
array = Matrix(
    [
        [m00, m01, m02, m03, m04, m05],
        [m10, m11, m12, m13, m14, m15],
        [m20, m21, m22, m23, m24, m25],
        [m30, m31, m32, m33, m34, m35],
        [m40, m41, m42, m43, m44, m45],
        [m50, m51, m52, m53, m54, m55]
    ]
)

array_right = Matrix(
    [
        [m13, m14, m15],
        [m23, m24, m25],
        [m33, m34, m35]
    ]
)

array_up = Matrix(
    [
        [m01, m02, m03],
        [m11, m12, m13],
        [m21, m22, m23]
    ]
)

array_left = Matrix(
    [
        [m20, m21, m22],
        [m30, m31, m32],
        [m40, m41, m42]
    ]
)

array_bottom = Matrix(
    [
        [m32, m33, m34],
        [m42, m43, m44],
        [m52, m53, m54]
    ]
)


# Define variables and equations for the system of equations
variables = (
    m00, m01, m02, m03, m04, m05,
    m10, m11, m12, m13, m14, m15,
    m20, m21, m22, m23, m24, m25,
    m30, m31, m32, m33, m34, m35,
    m40, m41, m42, m43, m44, m45,
    m50, m51, m52, m53, m54, m55,
    sr, su, sl, sb
)

equations_outside = [
    m00 - 0,
    m10 - 0,
    m04 - 0,
    m05 - 0,
    m50 - 0,
    m51 - 0,
    m45 - 0,
    m55 - 0
]

equations_initial = [
    m34 - a,
    m13 - b,
    m21 - c,
    m42 - d
]

equations_right = [
    sr - sum(array_right.row(0)),
    sr - sum(array_right.row(1)),
    sr - sum(array_right.row(2)),
    sr - sum(array_right.col(0)),
    sr - sum(array_right.col(1)),
    sr - sum(array_right.col(2)),
    sr - trace(array_right),
    sr - m33 - m24 - m15
]

equations_up = [
    su - sum(array_up.row(0)),
    su - sum(array_up.row(1)),
    su - sum(array_up.row(2)),
    su - sum(array_up.col(0)),
    su - sum(array_up.col(1)),
    su - sum(array_up.col(2)),
    su - trace(array_up),
    su - m21 - m12 - m03
]

equations_left = [
    sl - sum(array_left.row(0)),
    sl - sum(array_left.row(1)),
    sl - sum(array_left.row(2)),
    sl - sum(array_left.col(0)),
    sl - sum(array_left.col(1)),
    sl - sum(array_left.col(2)),
    sl - trace(array_left),
    sl - m40 - m31 - m22
]

equations_bottom = [
    sl - sum(array_bottom.row(0)),
    sl - sum(array_bottom.row(1)),
    sl - sum(array_bottom.row(2)),
    sl - sum(array_bottom.col(0)),
    sl - sum(array_bottom.col(1)),
    sl - sum(array_bottom.col(2)),
    sl - trace(array_bottom),
    sl - m52 - m43 - m34
]

equations = equations_initial + equations_right + equations_up + equations_left + equations_bottom


# solve system of equations
sol = linsolve(equations, variables) #next(iter(linsolve(equations, variables)))

"""
find inside of cross from outside of cross
"""

(aa, bb, cc, dd) = symbols('aa, bb, cc, dd')
(xx, yy, ww, zz) = symbols('xx, yy, ww, zz')

equations = [
    4*bb + yy - 2*xx - 3*aa,
    4*cc + ww - 2*yy - 3*bb,
    4*dd + zz - 2*ww - 3*cc,
    4*aa + xx - 2*zz - 3*dd
]

inside = next(iter(linsolve(equations, (dd, yy, ww, zz))))

for i in range(0,4):
    print("{} = {}".format(('dd', 'yy', 'ww', 'zz')[i], inside[i]))