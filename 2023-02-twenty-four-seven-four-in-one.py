# Imports
# ----------------------------------------------------------------------
from typing import Optional
from copy import deepcopy
from pprint import pprint


# Types
# ----------------------------------------------------------------------

Board = list[list[int]]
Choices = list[list[int]]


# Data
# ----------------------------------------------------------------------

n = -1
matrix = [
    [n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, n, 1, n, n, n, n, n, 6, 5, n],
    [n, n, n, n, n, 3, n, n, n, n, 6, n],
    [n, 4, n, n, n, n, n, n, 7, n, n, n],
    [n, n, n, n, 2, n, n, n, n, n, n, 7],
    [n, n, 6, n, n, n, n, n, 3, 7, n, n],
    [n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, n, 7, n, 5, n, n, n, n, n, n],
    [n, 5, n, n, n, 7, n, n, n, n, n, n],
    [n, 6, 7, n, n, n, n, n, n, n, n, n],
    [n, n, n, n, 6, n, n, n, n, n, n, n],
]

blue = [
    [5,  7,  7, 33, 29, 2, 40, 28, n, n, 36, n], # row
    [6,  n,  n,  4,  n, n,  n,  n, n, n,  n, 5], # col rev
    [4,  n,  n,  1,  n, n,  n,  n, n, n,  n, 7], # row rev
    [6, 36, 30, 34, 27, 3, 40, 27, n, n,  7, n], # col
]


# Functions
# ----------------------------------------------------------------------

# Bit functions

def count(num: int) -> int:
    ''' Number of 1s in binary representation of num. '''
    ans = 0
    while num:
        ans += 1
        num &= (num - 1)
    return ans


def first_bit(num: int) -> int:
    ''' Location of first 1 in binary representation of num. '''
    for i in range(8):
        if num & (1 << i):
            return i
    return -1


# Conversion between board and choices

def board(cm: Choices) -> Board:
    def board_aux(c: int) -> int:
        return first_bit(c) if count(c) == 1 else -1
    return [[board_aux(cm[i][j]) for j in range(12)] for i in range(12)]


def choices(xm: Board) -> Choices:
    def choices_aux(x: int) -> int:
        return 255 if x == -1 else 1 << x
    return [[choices_aux(xm[i][j]) for j in range(12)] for i in range(12)]


# Backtracking algorithm

def solution(xm: Board) -> Optional[Board]:
    stack = [choices(xm)]
    while stack:
        cm = prune(stack.pop())
        if reject(cm):
            continue
        if accept(cm):
            return board(cm)
        stack += expand(cm)
    return None


def accept(cm: Choices) -> bool:
    return all(count(cm[i][j]) == 1 for i in range(12) for j in range(12))


def reject(cm: Choices) -> bool:
    return blocked(cm) or not connected(cm)


def blocked(cm: Choices) -> bool:
    return any(cm[i][j] == 0 for i in range(12) for j in range(12))


def connected(cm: Choices) -> bool:
    direc = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    visit = {(1, 3)}
    stack = [(1, 3)]
    verts = sum(cm[i][j] != 1 for i in range(12) for j in range(12))

    while stack:
        i, j = stack.pop()
        verts -= 1
        
        for dx, dy in direc:
            x = i + dx
            y = j + dy
            if not (0 <= x < 12 and 0 <= y < 12):
                continue
            if cm[x][y] == 1:
                continue
            if (x, y) in visit:
                continue
            visit.add((x, y))
            stack.append((x, y))

    return not verts


def expand(cm: Choices) -> list[Choices]:
    for num_choices in range(2, 9):
        # find cell c = c[i][j] such that count(c) == num_choices
        for i in range(12):
            for j in range(12):
                c = cm[i][j]
                if count(c) == num_choices:
                    # return list of copies of cm, but with cm[i][j]
                    # replaced by each possible value
                    ans = []
                    for x in range(8):
                        if c & (1 << x):
                            cm_new = deepcopy(cm)
                            cm_new[i][j] = 1 << x
                            ans.append(cm_new)
                    return ans

    return []


def prune(cm: Choices) -> Choices:
    ans = deepcopy(cm)
    prune_again = True
    known_cells = sum(bool(cm[i][j]) for i in range(12) for j in range(12))

    while prune_again:

        # prune based on 2x2
        # --------------------------------------------------------------
        for i in range(11):
            for j in range(11):
                for corner in range(4):
                    x = i + ((corner + 1) % 4 > 1)
                    y = j + (corner > 1)

                    # if all cells except (x, y) do not have a 0, 
                    # then cell (x, y) can't be > 0.
                    remove = True
                    for other_corner in range(4):
                        if other_corner != corner:
                            x_ = i + ((other_corner + 1) % 4 > 1)
                            y_ = j + (other_corner > 1)
                            remove &= not (ans[x_][y_] & 1)

                    if remove:
                        ans[x][y] &= 1

        
        # prune based on one 1, ..., seven 7
        # --------------------------------------------------------------
        for grid in range(4):
            i0 = 5 if ((grid + 1) % 4 > 1) else 0
            i1 = 12 if ((grid + 1) % 4 > 1) else 7
            j0 = 5 if grid % 4 > 1 else 0
            j1 = 12 if grid % 4 > 1 else 7

            coordinates = {x: [] for x in range(8)}
            for i in range(i0, i1):
                for j in range(j0, j1):
                    if count(ans[i][j]) == 1:
                        coordinates[first_bit(ans[i][j])].append((i, j))

            for x in range(1, 8):
                # If there are x or more cells with only possibility x,
                # x is not in the other cells
                if len(coordinates[x]) >= x:
                    for i in range(i0, i1):
                        for j in range(j0, j1):
                            if (i, j) not in coordinates[x][:x]:
                                ans[i][j] &= (1 << x) ^ 255


        # prune based on row/col sum = 20, 4 nums
        # --------------------------------------------------------------
        for grid in range(4):
            i0 = 5 if ((grid + 1) % 4 > 1) else 0
            i1 = 12 if ((grid + 1) % 4 > 1) else 7
            j0 = 5 if grid % 4 > 1 else 0
            j1 = 12 if grid % 4 > 1 else 7

            rows = [[(i, j) for j in range(j0, j1)] for i in range(i0, i1)]
            cols = [[(i, j) for i in range(i0, i1)] for j in range(j0, j1)]
            arrs = rows + cols

            for arr in arrs:
                cms = [ans[i][j] for (i, j) in arr]
                
                nk0 = sum(first_bit(c) == 0 for c in cms if count(c) == 1)
                nk1 = sum(first_bit(c) >= 1 for c in cms if count(c) == 1)
                n = 4 - nk1
                s = 20 - sum(first_bit(c) for c in cms if count(c) == 1)

                if not (nk0 <= 3 and nk1 <= 4) or (n == 0 and s != 0):
                    return [[0 for _ in range(12)] for _ in range(12)]
                
                # loop over unknown cells
                for (i, j), c in zip(arr, cms):
                    if count(c) > 1:
                        # loop over x s.t. x is not part of partition
                        for x in range(1, 8):
                            if n and not (n - 1 <= s - x <= 7*(n - 1)):
                                ans[i][j] &= (1 << x) ^ 255

        
        # prune based on bluenum sum
        # --------------------------------------------------------------
        rows = [(blue[0][i], [(i, j) for j in range(5, 7)]) for i in range(12)]
        cols = [(blue[3][j], [(i, j) for i in range(5, 7)]) for j in range(12)]
        arrs = rows + cols

        for bluenum, arr in arrs:
            if bluenum <= 7:
                continue
                
            cms = [ans[i][j] for (i, j) in arr]
            n = sum(count(c) != 1 for c in cms)
            s = 40 - bluenum - sum(first_bit(c) for c in cms if count(c) == 1)

            if n == 0 and s != 0:
                return [[0 for _ in range(12)] for _ in range(12)]
            
            # loop over unknown cells
            for (i, j), c in zip(arr, cms):
                if count(c) > 1:
                    # loop over x s.t. x is not part of partition
                    for x in range(1, 8):
                        if n and not (0 <= s - x <= 7*(n - 1)):
                            ans[i][j] &= (1 << x) ^ 255


        # prune based on bluenum line of sight
        # --------------------------------------------------------------
        for side in range(4):
            for a in range(12):
                bluenum = blue[side][a]
                if 1 <= bluenum <= 7:
                    for b in range(12):
                        i = a if not side % 2 else (b if side == 3 else 11 - b)
                        j = a if side % 2 else (b if not side else 11 - b)
                        if ans[i][j] != 1:
                            # remove all except {0, bluenum} from ans[i][j]
                            for x in range(1, 8):
                                if x != bluenum:
                                    ans[i][j] &= (1 << x) ^ 255
                            break

        # Check if we should prune again
        newkn_cells = sum(bool(cm[i][j]) for i in range(12) for j in range(12))
        prune_again = known_cells != newkn_cells
        known_cells = newkn_cells

    return ans


# Solution
# ----------------------------------------------------------------------

sol = solution(matrix)
print(f'\nsolution = ')
pprint(sol)
'''
solution = 
    [0 5 0 6 0 3 6 0 0 0 7 4]
    [0 7 7 1 0 0 5 0 4 6 5 0]
    [0 0 0 7 5 3 5 0 6 0 6 0]
    [6 4 3 0 0 7 0 5 7 1 0 0]
    [7 0 0 0 2 7 4 2 0 0 0 7]
    [2 0 6 6 6 0 0 6 3 7 0 4]
    [5 4 4 0 7 0 0 7 0 6 2 5]
    [7 0 0 0 1 5 7 1 0 0 7 0]
    [0 5 3 7 0 5 0 0 5 0 4 6]
    [6 5 0 0 0 7 2 0 6 0 0 5]
    [0 6 7 3 0 0 4 6 6 4 0 0]
    [0 0 0 4 6 3 7 0 0 3 7 0]
'''


# Product of areas
# ----------------------------------------------------------------------

def prareas(xm: Board) -> int:
    ans = 1
    direc = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    verts = {(i, j) for i in range(12) for j in range(12) if not xm[i][j]}

    while verts:
        i0, j0 = verts.pop()
        curr = 1
        stack = [(i0, j0)]
        
        while stack:
            i, j = stack.pop()
            
            for dx, dy in direc:
                x = i + dx
                y = j + dy
                if not (0 <= x < 12 and 0 <= y < 12):
                    continue
                if xm[x][y]:
                    continue
                if (x, y) not in verts:
                    continue
                verts.remove((x, y))
                stack.append((x, y))
                curr += 1

        ans *= curr

    return ans

print(f'\nans = {prareas(sol)}') # type: ignore
# ans = 74649600