from math import factorial, comb
from fractions import Fraction
from sympy.abc import a, b, c, d, e
from sympy import Poly


'''
This solution is based on the math.stackexchange post
https://math.stackexchange.com/questions/3934750/how-to-solve-this-candy-collectors-puzzle

p = num/den, where

    den = # ways of sorting candy
        = (5**2)! / (5!)**5

    num = # ways of sorting candy, s.t. every child owns more candy of
            one type than every other child
        = 5! * count

    where
    count = # ways of sorting candy, s.t.
            child 1 has the most of candy 1
            child 2 has the most of candy 2
            child 3 has the most of candy 3
            child 4 has the most of candy 4
            child 5 has the most of candy 5


To compute count, we will use generating functions. We will define a
polynomial f1(a, b, c, d, e) where
    - a, b, c, d, e represent the 5 children
    - exponents represent how many of candy 1 each child has

Let f1(a, b, c, d, e) be the polynomial whose coefficient of
a^i b^j c^k d^l e^m is
    - 0 if i + j + k + l + m != 5
    - 0 if i <= max{j, k, l, m}
    - if i + j + k + l + m == 5 and i > max{j, k, l, m}, then
      coef = # different ways of giving
               i of candy 1 to child a
               j of candy 1 to child b
               k of candy 1 to child c
               l of candy 1 to child d
               m of candy 1 to child e

I.e., f1 is given by the formula implemented in code below.
Define polys for the other types of candy f2, f3, f4, f5 analogously.

Then, count = coef of (abcde)^5 of f1*f2*f3*f4*f5
'''


def f(a, b, c, d, e):
    T5 = comb(5, 5) * a**5
    T4 = comb(5, 4) * a**4 * (b + c + d + e)
    T3 = comb(5, 3) * a**3 * ((b**2 + c**2 + d**2 + e**2)
                              + 2 * (b*c + b*d + b*e + c*d + c*e + d*e))
    T2 = comb(5, 2) * a**2 * factorial(3) * (c*d*e + b*d*e + b*c*e + b*c*d)
    return T5 + T4 + T3 + T2


f1 = Poly(f(a, b, c, d, e))
f2 = Poly(f(b, c, d, e, a))
f3 = Poly(f(c, d, e, a, b))
f4 = Poly(f(d, e, a, b, c))
f5 = Poly(f(e, a, b, c, d))
gf = f1*f2*f3*f4*f5

count = gf.coeff_monomial((a*b*c*d*e)**5)
den = Fraction(factorial(5**2), (factorial(5))**5)
num = factorial(5) * count
p = Fraction(num, den)
print(f'p = {p}')
# p = 318281087/8016470462
