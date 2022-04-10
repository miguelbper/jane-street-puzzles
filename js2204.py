import numpy as np
import pandas as pd
from itertools import permutations
from tqdm import tqdm
from functools import partial
from operator import add


''' Function that computes entire board from x, y, w, z '''

def board_from_four(x, y, w, z):
    
    # compute elements in the cross
    # if division has no remainder, the four squares will be magic 
    a = (-2*x +  9*y + 12*w + 16*z) // 35
    b = (16*x -  2*y +  9*w + 12*z) // 35
    c = (12*x + 16*y -  2*w +  9*z) // 35
    d = ( 9*x + 12*y + 16*w -  2*z) // 35

    # Empty entries
    m00 = None
    m10 = None
    m04 = None
    m05 = None
    m50 = None
    m51 = None
    m45 = None
    m55 = None

    # cross
    m33 = x
    m23 = y
    m22 = w
    m32 = z

    m34 = a
    m13 = b
    m21 = c
    m42 = d

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


def map_none(function, array):
    def function_none(x):
        return None if x == None else function(x)
    function_vectorized = np.vectorize(function_none)
    return function_vectorized(array)


def make_minimum(l):
    def make_minimum_l(ar):
        m = min(ar[ar!=None])
        return map_none(partial(add, l - m), ar)
    return make_minimum_l


''' Functions to test if the resulting board satisfies all conditions '''


# func: entire board -> int, sum
def sum_of(ar):
    return sum(ar[ar!=None])


# func: entire board -> bool, no duplicate numbers
def num_duplicates(ar):
    ar_f = ar[ar!=None]
    return len(ar_f) - len(np.unique(ar_f))


''' Functions to confirm that the result is correct '''


# func: entire board -> bool, all numbers positive
def all_numbers_positive(ar):
    return all(map(lambda x: x > 0, ar[ar!=None]))


# func: entire board -> bool, is magic/almost magic
# note: every board coming from the board_from_four function should satisfy this condition 
# as long as the divisions in board_from_four have no remainder. we check that this is the case before calling the function
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


def is_almost_magic_square(ar):
    sums = list(set((
        sum(ar[0,:]),
        sum(ar[1,:]),
        sum(ar[2,:]),
        sum(ar[:,0]),
        sum(ar[:,1]),
        sum(ar[:,2]),
        ar.trace(),
        ar[2,0] + ar[1,1] + ar[0,2]
    )))

    if len(sums) == 1:
        return True
    elif len(sums) == 2:
        return abs(sums[1] - sums[0]) == 1
    else:
        return False


def is_almost_magic(ar):
    return all(
        [
            is_almost_magic_square(square_right(ar)),
            is_almost_magic_square(square_up(ar)),
            is_almost_magic_square(square_left(ar)),
            is_almost_magic_square(square_bottom(ar))
        ]
    )

# func: entire board -> list, sorted list of elements of array
def sorted_list(ar):
    return sorted(ar[ar!=None])


''' Functions that define an iterator '''

def x_is_min(x, y, w, z):
    return x == min(x, y, w, z)


def abcd_are_ints(x, y, w, z):
    return all([
        (-2*x +  9*y + 12*w + 16*z) % 35 == 0,
        (16*x -  2*y +  9*w + 12*z) % 35 == 0,
        (12*x + 16*y -  2*w +  9*z) % 35 == 0,
        ( 9*x + 12*y + 16*w -  2*z) % 35 == 0
    ])


def is_valid_tuple(tup):
    (x, y, w, z) = tup
    return x_is_min(x, y, w, z) and abcd_are_ints(x, y, w, z)


def iterator(n):
    return filter(is_valid_tuple, permutations(range(1,n), 4))


''' Function that computes the magic square with the lowest sum such that x,y,w,z < n '''


def lowest_sum_grid(d, n):
    final_grid = board_from_four(0, 0, 0, 0)
    final_sum = 9999
    
    for (x, y, w, z) in tqdm(iterator(n)):
        grid = board_from_four(x, y, w, z)
        grid = make_minimum(1)(grid)
        s = sum_of(grid)

        if num_duplicates(grid) <= d and s < final_sum:
            final_grid = grid
            final_sum = s

    return final_grid


def lowest_sum_grid_divide(d, p, n):
    final_grid = board_from_four(0, 0, 0, 0)
    final_sum = 9999
    
    for (x, y, w, z) in tqdm(iterator(n)):
        grid = board_from_four(x, y, w, z)
        grid = make_minimum(p)(grid)
        grid = map_none(lambda x: x // p, grid)
        s = sum_of(grid)

        if num_duplicates(grid) <= d and s < final_sum:
            final_grid = grid
            final_sum = s

    return final_grid


def lowest_sum_grid_divide_alt(d, p, n):
    final_grid = board_from_four(0, 0, 0, 0)
    final_sum = 9999
    
    for (x, y, w, z) in tqdm(iterator(n)):
        grid = board_from_four(x, y, w, z)
        grid = map_none(lambda x: x // p, grid)
        grid = make_minimum(1)(grid)
        s = sum_of(grid)

        if num_duplicates(grid) <= d and s < final_sum:
            final_grid = grid
            final_sum = s

    return final_grid


def test_results(grid):
    return '''magic square = 
{m}

sum = {s}
is almost magic = {a}
no duplicates = {d}
only positive numbers = {p}

sorted flattened array = {f}
sum of sorted flattened array = {fs}
length sorted flattened array = {fl}

list to submit = {ll}'''.format(
        m = pd.DataFrame(grid).to_string(header=False, index=False),
        s = sum_of(grid),
        a = is_almost_magic(grid),
        d = num_duplicates(grid) == 0,
        p = all_numbers_positive(grid),
        f = sorted_list(grid),
        fs = sum(sorted_list(grid)),
        fl = len(sorted_list(grid)),
        ll = list(grid[grid!=None])
    )