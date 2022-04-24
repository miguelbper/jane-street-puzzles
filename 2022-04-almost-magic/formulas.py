import sympy as sp

(au, bu, u) = sp.symbols("au, bu, u")
(al, bl, l) = sp.symbols("al, bl, l")
(ad, bd, d) = sp.symbols("ad, bd, d")
(ar, br, r) = sp.symbols("ar, br, r")

# any magic square is of the following form, for some a, b, c
def magic(a, b, c):
    return sp.Matrix([[c     - b, c + a + b, c - a    ],
                      [c - a + b, c        , c + a - b],
                      [c + a    , c - a - b, c     + b]])

def rotate(n, m):
    n = n % 4
    for i in range(0,n):
        m = sp.Matrix([m.col(2).transpose(), 
                       m.col(1).transpose(), 
                       m.col(0).transpose()])
    return m

def board(mu, ml, md, mr):
    mu = rotate(0, mu)
    ml = rotate(1, ml)
    md = rotate(2, md)
    mr = rotate(3, mr)

    m = sp.zeros(6)

    m[0:3,1:4] = mu
    m[2:5,0:3] = ml
    m[3:6,2:5] = md
    m[1:4,3:6] = mr

    return m

# consider 4 magic squares as above
mu = magic(au, bu, u)
ml = magic(al, bl, l)
md = magic(ad, bd, d)
mr = magic(ar, br, r)

# the overlapping entries must agree
e0 = mu[1,2] - mr[2,0]
e1 = mu[2,0] - ml[1,2]
e2 = mu[2,1] - ml[2,2]
e3 = mu[2,2] - mr[2,1]
e4 = ml[2,1] - md[2,2]
e5 = mr[2,2] - md[2,1]
e6 = mr[1,2] - md[2,0]
e7 = ml[2,0] - md[1,2]

# solve the following system of linear equations
# the variables will then be written as a linear combination of the parameters u, l, d, r
eqs = [e0, e1, e2, e3, e4, e5, e6, e7]
vrs = [au, bu, al, bl, ad, bd, ar, br]
sol = next(iter(sp.linsolve(eqs, vrs)))

(au, bu, al, bl, ad, bd, ar, br) = tuple(sol)

mu = magic(au, bu, u)
ml = magic(al, bl, l)
md = magic(ad, bd, d)
mr = magic(ar, br, r)

m = board(mu, ml, md, mr)

print("au = {}".format(au))
print("bu = {}".format(bu))
print("m = {}".format(m))