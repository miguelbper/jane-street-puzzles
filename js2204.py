import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm

""" Functions to compute entire board from a few cells """


# func: 1 number -> 4 numbers, non canonical


# func: 4 numbers -> entire board
def board_from_four(a, b, c, x):
    
    # Empty entries
    m00 = None
    m10 = None
    m04 = None
    m05 = None
    m50 = None
    m51 = None
    m45 = None
    m55 = None

    # Four given entries in cross
    m34 = a
    m13 = b
    m21 = c
    m33 = x

    # Four other entries in cross
    m42 = 4*a - 4*b - 2*c + 3*x
    m23 = 3*m34 + 2*m33 - 4*m13
    m22 = 3*m13 + 2*m23 - 4*m21
    m32 = 3*m21 + 2*m22 - 4*m42 

    # Right square
    sl = m13 + m23 + m33
    m35 = sl - m33 - m34
    m24 = sl - m13 - m35
    m14 = sl - m24 - m34
    m25 = sl - m23 - m24
    m15 = sl - m13 - m14

    # Up square
    su = m21 + m22 + m23
    m03 = su - m13 - m23
    m12 = su - m21 - m03
    m11 = su - m12 - m13
    m02 = su - m12 - m22
    m01 = su - m02 - m03 

    # Left square
    sr = m22 + m32 + m42
    m20 = sr - m21 - m22
    m31 = sr - m20 - m42
    m30 = sr - m31 - m32
    m41 = sr - m31 - m21
    m40 = sr - m41 - m42

    # Bottom square
    sb = m32 + m33 + m34
    m52 = sb - m32 - m42
    m43 = sb - m52 - m34
    m53 = sb - m33 - m43
    m44 = sb - m42 - m43
    m54 = sb - m52 - m53 

    # Define board
    return np.array(
        [
            [m00, m01, m02, m03, m04, m05],
            [m10, m11, m12, m13, m14, m15],
            [m20, m21, m22, m23, m24, m25],
            [m30, m31, m32, m33, m34, m35],
            [m40, m41, m42, m43, m44, m45],
            [m50, m51, m52, m53, m54, m55]
        ]
    )


""" Functions to test if the resulting board satisfies all conditions """


# func: entire board -> int, sum
def sum_of(ar):
    return sum(ar[ar!=None])


# func: entire board -> bool, all numbers positive
def all_numbers_positive(ar):
    return all(map(lambda x: x > 0, ar[ar!=None]))


# func: entire board -> bool, no duplicate numbers
def no_duplicate_numbers(ar):
    ar_f = ar[ar!=None]
    return len(np.unique(ar_f)) == len(ar_f)


# func: entire board -> bool, is magic/almost magic
# note: every board coming from the board_from_four function should satisfy this condition
def square_right(ar):
    return ar[1:4, 3:]

def square_up(ar):
    return ar[0:3, 1:4]

def square_left(ar):
    return ar[2:5, 0:3]

def square_bottom(ar):
    return ar[3:, 2:5]

def is_magic_square(ar):
    s = ar.trace()

    return all(
        [
            s == sum(ar[0,:]),
            s == sum(ar[1,:]),
            s == sum(ar[2,:]),
            s == sum(ar[:,0]),
            s == sum(ar[:,1]),
            s == sum(ar[:,2]),
            s == ar[2,0] + ar[1,1] + ar[0,2]
        ]
    )

def is_magic(ar):
    return all(
        [
            is_magic_square(square_right(ar)),
            is_magic_square(square_up(ar)),
            is_magic_square(square_left(ar)),
            is_magic_square(square_bottom(ar))
        ]
    )


""" Loop that finds a magic square"""

iterator = product(range(0,35), repeat = 4)

final_grid = board_from_four(0, 0, 0, 0)
final_sum = 9999
updated = False

for (a, b, c, x) in tqdm(iterator):
    grid = board_from_four(a, b, c, x)
    s = sum_of(grid)
    p = all_numbers_positive(grid)
    nd = no_duplicate_numbers(grid)

    if p and nd and s < final_sum:
        final_grid = grid
        final_sum = s
        updated = True

print(final_grid)
print(final_sum)
print(updated)