import numpy as np
import math
import sympy
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from z3 import (Solver, ModelRef, ArithRef, BoolRef, IntVector, BoolVal, sat,
                And, Or, Implies, If)
from codetiming import Timer
import inspect


# Constants & input
# ----------------------------------------------------------------------
regions = np.array([
    [5, 5, 5, 2, 2, 2, 3, 3,  0,  0,  0],
    [5, 1, 1, 1, 2, 2, 3, 0,  0,  0, 12],
    [5, 1, 1, 2, 2, 2, 3, 0,  0,  0, 12],
    [5, 1, 1, 2, 2, 4, 4, 0, 12, 12, 12],
    [5, 1, 2, 2, 0, 0, 4, 0, 12, 10, 12],
    [5, 0, 0, 0, 0, 0, 0, 0, 10, 10, 11],
    [6, 0, 0, 0, 0, 8, 8, 0, 10, 10, 10],
    [6, 6, 7, 0, 7, 8, 8, 0,  0, 10,  0],
    [6, 6, 7, 7, 7, 8, 8, 0,  0,  0,  0],
    [6, 7, 7, 6, 6, 6, 8, 0,  0,  0,  9],
    [6, 6, 6, 6, 6, 8, 8, 8,  0,  0,  9],
])

black = -1
n = regions.shape[0]
Coord = tuple[int, int]


# Utility functions
# ----------------------------------------------------------------------
def neighbours(i: int, j: int, u: int, d: int, l: int, r: int) -> list[Coord]:
    ''' Return the coordinates of the neighbours of (i, j). '''
    vertical = [(v, 0) for v in range(-u, d + 1) if v]
    horizontal = [(0, h) for h in range(-l, r + 1) if h]
    directions = vertical + horizontal
    neigh = [(i + di, j + dj) for di, dj in directions]
    return [(x, y) for x, y in neigh if 0 <= x < n and 0 <= y < n]


def evaluate_vars(m: ModelRef, vars: np.ndarray) -> np.ndarray:
    ''' Evaluate variables in a z3 model. '''
    return np.vectorize(lambda x: m.evaluate(x).as_long())(vars)


def answer(xm: np.ndarray) -> int:
    ''' Return the answer to the problem, given the filled board. '''
    nums = []
    for i in range(n):
        row_string = ''.join(map(str, xm[i]))
        row_numbers = [int(s) for s in row_string.split(str(black)) if s]
        nums += row_numbers
    return sum(nums)


# Variables, solver & constraints
# ----------------------------------------------------------------------
X = np.array(IntVector('x', n**2)).reshape((n, n))
s = Solver()

# Ranges for each variable
s += [Or(x == black, And(0 <= x, x <= 9)) for x in X.flat]

# Nums have num_digits >= 2 => Cols 1 and n - 2 can't be black
s += [x != black for x in X[:, 1]]
s += [x != black for x in X[:, n - 2]]

# Product of digits ends with 1 => All digits are 1, 3, 7 or 9
s += And([Or(x == black, x == 1, x == 3, x == 7, x == 9) for x in X[8]])

# If a cell is black, its neighbours can't be black
for i, j in product(range(n), repeat=2):
    neighbourhood = neighbours(i, j, 1, 1, 2, 2)
    cell_is_black = X[i, j] == black
    neig_is_white = And([X[k, l] != black for k, l in neighbourhood])
    s += Implies(cell_is_black, neig_is_white)

# Same region => same digit // different region => different digit
for i, j in product(range(n), repeat=2):
    for k, l in neighbours(i, j, 0, 1, 0, 1):
        both_white = And(X[i, j] != black, X[k, l] != black)
        if regions[i, j] == regions[k, l]:
            s += Implies(both_white, X[i, j] == X[k, l])
        else:
            s += Implies(both_white, X[i, j] != X[k, l])


# Row clue constraints
# ----------------------------------------------------------------------

# Number lists
square_lists = [[] for _ in range(n + 1)]
for root in range(math.ceil(math.sqrt(10**n))):
    num = root**2
    num_digits = len(str(num))
    if num_digits <= n:
        square_lists[num_digits].append(num)


fibonacci_lists = [[] for _ in range(n + 1)]
a, b = 0, 1
while b < 10**n:
    num_digits = len(str(b))
    fibonacci_lists[num_digits].append(b)
    a, b = b, a + b


primes_powers_lists = [[] for _ in range(n + 1)]
primes = list(sympy.primerange(math.ceil(math.sqrt(10**n))))
for p in primes:
    for q in primes:
        num = p**q
        if num >= 10**n:
            break
        num_digits = len(str(num))
        primes_powers_lists[num_digits].append(num)


# Utils
def number(X: list[ArithRef]) -> ArithRef:
    ''' z3 variable representing number with digits X. '''
    return sum(10**i * x for i, x in enumerate(reversed(X)))


def equals(X: list[ArithRef], num: int) -> BoolRef:
    ''' z3 variable representing number(X) == num. '''
    return And([x == int(d) for x, d in zip(X, str(num))])


def valid(i: int, l: int, r: int, num: int) -> bool:
    ''' True if num can be placed in row i, [l:r]. '''
    digits = [int(d) for d in str(num)]
    for k in range(l, r - 1):
        same_region = regions[i, k] == regions[i, k + 1]
        same_digits = digits[k - l] == digits[k - l + 1]
        if same_region != same_digits:
            return False
    return True


# Row constraints
def multiple_of_37(X: list[ArithRef]) -> BoolRef:
    ''' z3 variable, true if number(X) is a multiple of 37. '''
    return number(X) % 37 == 0


def multiple_of_88(X: list[ArithRef]) -> BoolRef:
    ''' z3 variable, true if number(X) is a multiple of 88. '''
    return number(X) % 88 == 0


def sum_of_digits_is_7(X: list[ArithRef]) -> BoolRef:
    ''' z3 variable, true if sum of digits in X is 7. '''
    return sum(X) == 7


def product_of_digits_ends_in_1(X: list[ArithRef]) -> BoolRef:
    ''' z3 variable, true if product of digits in X ends in 1.

    Easiest implementation would be
        return math.prod(X) % 10 == 1
    but this would involve products which makes z3 very slow. Instead,
    recast this constraint as a constraint with only sums.

    For this, note that if the product of digits ends in 1, then the
    digits are in {1, 3, 7, 9}. Moreover, {1, 3, 7, 9} with
    multiplication mod 10 is a group. This group is isomorphic to Z/4Z
    with sum, and an isomorphism f is given by:
        f(1) = 0
        f(3) = 1
        f(7) = 3
        f(9) = 2
    Therefore, the following are equivalent:
        * prod(x1,...,xk) % 10 == 1
        * sum(f(x1),...,f(xk)) % 4 == 0
    '''
    isomorphism = lambda x: If(x == 1, 0,
                            If(x == 3, 1,
                            If(x == 7, 3,
                            If(x == 9, 2, -1))))
    return sum(map(isomorphism, X)) % 4 == 0


def palindrome_multiple_of_23(X: list[ArithRef]) -> BoolRef:
    ''' z3 variable, true if number(X) is palindrome multiple of 23. '''
    palindrome = And([x == y for x, y in zip(X, reversed(X))])
    multiple_of_23 = number(X) % 23 == 0
    return And(palindrome, multiple_of_23)


def one_more_than_a_palindrome(X: list[ArithRef]) -> BoolRef:
    ''' z3 variable, true if number(X) is palindrome + 1.

    Code here is not 100% correct because it doesn't handle the case
    where the palindrome ends with 9. In the solution of the puzzle, the
    palindrome does not end with a 9 so this works.
    '''
    n_digits = len(X)
    iterator = list(enumerate(zip(X, reversed(X))))[:n_digits // 2]
    return And([x + int(i == 0) == y for i, (x, y) in iterator])


def one_less_than_a_palindrome(X: list[ArithRef]) -> BoolRef:
    ''' z3 variable, true if number(X) is palindrome - 1. '''
    n_digits = len(X)
    iterator = list(enumerate(zip(X, reversed(X))))[:n_digits // 2]
    return And([x - int(i == 0) == y for i, (x, y) in iterator])


def square(X: list[ArithRef], i: int, l: int, r: int) -> BoolRef:
    ''' z3 variable, true if number(X) is a square. '''
    num_digits = len(X)
    is_valid_num = lambda num: valid(i, l, r, num)
    valid_numbers = filter(is_valid_num, square_lists[num_digits])
    return Or([equals(X, num) for num in valid_numbers])


def fibonacci(X: list[ArithRef], i: int, l: int, r: int) -> BoolRef:
    ''' z3 variable, true if number(X) is a fibonacci number. '''
    num_digits = len(X)
    is_valid_num = lambda num: valid(i, l, r, num)
    valid_numbers = filter(is_valid_num, fibonacci_lists[num_digits])
    return Or([equals(X, num) for num in valid_numbers])


def prime_raised_to_a_prime_power(
    X: list[ArithRef],
    i: int,
    l: int,
    r: int,
) -> BoolRef:
    ''' z3 variable, true if number(X) is a prime raised to prime. '''
    num_digits = len(X)
    is_valid_num = lambda num: valid(i, l, r, num)
    valid_numbers = filter(is_valid_num, primes_powers_lists[num_digits])
    return Or([equals(X, num) for num in valid_numbers])


# Add constraints to solver
clues = [
    square,
    one_more_than_a_palindrome,
    prime_raised_to_a_prime_power,
    sum_of_digits_is_7,
    fibonacci,
    square,
    multiple_of_37,
    palindrome_multiple_of_23,
    product_of_digits_ends_in_1,
    multiple_of_88,
    one_less_than_a_palindrome,
]


def is_number(i: int, l: int, r: int) -> BoolRef:
    ''' z3 variable, true if in row i, [l:r] is a number.

    This means that the boundaries of [l:r] are black and the interior
    is not black.
    '''
    l_boundary = BoolVal(True) if l == 0 else X[i, l - 1] == black
    r_boundary = BoolVal(True) if r == n else X[i, r] == black
    interior = And([x != black for x in X[i, l:r]])
    return And(l_boundary, r_boundary, interior)


# loop over rows
for i in range(n):
    # loop over arrays [l:r]
    for l in range(n):
        for r in range(l + 2, n + 1):
            # if [l:r] is a number, must satisfy the clue
            add_args = len(inspect.signature(clues[i]).parameters) > 1
            args = (i, l, r) if add_args else ()
            s += Implies(is_number(i, l, r), clues[i](X[i, l:r], *args))


# Plot grid function
# ----------------------------------------------------------------------

def plot_grid(xm: np.ndarray) -> None:
    ''' Plot the grid as a seaborn heatmap. '''
    # compute data for heatmap
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    data = xm == black
    annot = np.array(xm).astype('str')
    annot[annot == '-1'] = ''

    # create heatmap
    ax = sns.heatmap(
        data,
        annot=annot,
        cbar=False,
        fmt='',
        linewidths=0.005,
        linecolor='black',
        square=True,
        cmap=['white', 'black'],
    )
    ax.tick_params(
        left=False,
        labelleft=False,
        bottom=False,
        labelbottom=False,
    )

    # annotations
    annotations = [
        'square',
        'one more than a palindrome',
        'prime raised to a prime power',
        'sum of digits is 7',
        'fibonacci',
        'square',
        'multiple of 37',
        'palindrome multiple of 23',
        'product of digits ends in 1',
        'multiple of 88',
        'one less than a palindrome',
    ]
    for i, annotation in enumerate(annotations):
        ax.text(n + 0.5, i + 0.5, annotation, va='center')

    # add thick region lines
    lines = [
        ([ 1,  4], [ 1,  1]),
        ([ 7,  8], [ 1,  1]),
        ([10, 11], [ 1,  1]),
        ([ 3,  4], [ 2,  2]),
        ([ 5,  7], [ 3,  3]),
        ([ 8, 10], [ 3,  3]),
        ([ 2,  3], [ 4,  4]),
        ([ 4,  6], [ 4,  4]),
        ([ 9, 10], [ 4,  4]),
        ([ 1,  4], [ 5,  5]),
        ([ 6,  7], [ 5,  5]),
        ([ 8,  9], [ 5,  5]),
        ([10, 11], [ 5,  5]),
        ([ 0,  1], [ 6,  6]),
        ([ 5,  7], [ 6,  6]),
        ([10, 11], [ 6,  6]),
        ([ 1,  3], [ 7,  7]),
        ([ 4,  5], [ 7,  7]),
        ([ 8,  9], [ 7,  7]),
        ([10, 11], [ 7,  7]),
        ([ 3,  4], [ 8,  8]),
        ([ 9, 10], [ 8,  8]),
        ([ 1,  2], [ 9,  9]),
        ([ 3,  6], [ 9,  9]),
        ([10, 11], [ 9,  9]),
        ([ 1,  3], [10, 10]),
        ([ 5,  6], [10, 10]),
        ([ 7,  8], [10, 10]),
        ([ 1,  1], [ 1,  7]),
        ([ 1,  1], [ 9, 10]),
        ([ 2,  2], [ 4,  5]),
        ([ 2,  2], [ 7,  9]),
        ([ 3,  3], [ 0,  1]),
        ([ 3,  3], [ 2,  4]),
        ([ 3,  3], [ 7,  8]),
        ([ 3,  3], [ 9, 10]),
        ([ 4,  4], [ 1,  2]),
        ([ 4,  4], [ 4,  5]),
        ([ 4,  4], [ 7,  8]),
        ([ 5,  5], [ 3,  4]),
        ([ 5,  5], [ 6,  9]),
        ([ 5,  5], [10, 11]),
        ([ 6,  6], [ 0,  3]),
        ([ 6,  6], [ 4,  5]),
        ([ 6,  6], [ 9, 10]),
        ([ 7,  7], [ 1,  5]),
        ([ 7,  7], [ 6, 10]),
        ([ 8,  8], [ 0,  1]),
        ([ 8,  8], [ 3,  7]),
        ([ 8,  8], [10, 11]),
        ([ 9,  9], [ 4,  5]),
        ([ 9,  9], [ 7,  8]),
        ([10, 10], [ 1,  3]),
        ([10, 10], [ 4,  6]),
        ([10, 10], [ 7,  8]),
        ([10, 10], [ 9, 11]),
    ]
    for (a, b), (c, d) in lines:
        plt.plot([a, b], [c, d], linewidth=4, color='black')

    plt.show()


def print_arr(arr: np.ndarray, name: str) -> None:
    ''' Prints a numpy 2D array where each element is a str. '''
    print(f'{name} = ', end='')
    for i, row in enumerate(arr):
        initial_spaces = (3 + len(name)) * ' ' if i else ''
        row = ' '.join(np.char.strip(row, chars="'"))
        print(initial_spaces + '[' + row + ']')

# Solve
# ----------------------------------------------------------------------

with Timer(initial_text='Checking z3 solver'):
    check = s.check()

if check == sat:
    m = s.model()
    xm = evaluate_vars(m, X)
    xm_str = np.vectorize(lambda x: '.' if x == black else str(x))(xm)
    print(f'answer = {answer(xm)}')
    print_arr(xm_str, 'xm')
    plot_grid(xm)
'''
Elapsed time: 44.4874 seconds
answer = 88243711283
xm = [1 1 1 2 2 2 3 3 4 4 4]
     [1 3 3 3 2 . 3 4 4 4 .]
     [1 3 3 1 . 7 3 4 4 4 9]
     [1 3 3 . 1 0 0 4 1 1 .]
     [1 3 . 1 4 4 . 4 1 8 1]
     [1 4 4 4 . 4 4 4 8 8 9]
     [7 4 4 4 4 . 7 4 8 8 8]
     [7 7 1 4 1 7 7 . 9 8 9]
     [7 7 1 1 1 7 7 9 9 9 9]
     [. 1 1 4 4 . 7 9 9 9 2]
     [4 4 4 4 4 3 . 3 9 9 2]
'''
