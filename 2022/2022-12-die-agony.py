# types
# ----------------------------------------------------------------------

Matrix = list[list[int]]
Coords = tuple[int, int]
Visited = list[Coords]
MInt = int | None
Die = tuple[MInt, MInt, MInt, MInt, MInt, MInt]
State = tuple[int, Coords, Die, Visited]


# data
# ----------------------------------------------------------------------

# fmt: off
grid = [
    [ 57,  33, 132, 268, 492, 732],
    [ 81, 123, 240, 443, 353, 508],
    [186,  42, 195, 704, 452, 228],
    [ -7,   2, 357, 452, 317, 395],
    [  5,  23,  -4, 592, 445, 620],
    [  0,  77,  32, 403, 337, 452],
]
# fmt: on

m = len(grid)

n0 = 0
x0 = m - 1
y0 = 0
c0 = (x0, y0)
die0 = (None,) * 6
vis0 = [c0]
initial = (n0, c0, die0, vis0)


# functions
# ----------------------------------------------------------------------


def final(state: State) -> bool:
    # return if state is final, i.e. coord = top right
    return state[1] == (0, m - 1)


def tilt(dir: Coords, die: Die) -> Die:
    top, bottom, up, down, left, right = die

    if dir == (0, 1):  # tilt up
        return (
            down,
            up,
            top,
            bottom,
            left,
            right,
        )
    if dir == (0, -1):  # tilt down
        return (
            up,
            down,
            bottom,
            top,
            left,
            right,
        )
    if dir == (-1, 0):  # tilt left
        return (
            right,
            left,
            up,
            down,
            top,
            bottom,
        )
    if dir == (1, 0):  # tilt right
        return (
            left,
            right,
            up,
            down,
            bottom,
            top,
        )
    return die


def move(direction: Coords, state: State) -> State | None:
    n, c, die, vis = state
    x, y = c
    a, b = direction

    n_ = n + 1
    x_ = x + a
    y_ = y + b
    c_ = (x_, y_)
    vis_ = vis + [c_]

    if not (0 <= x_ < m and 0 <= y_ < m):
        return None

    die_ = tilt(direction, die)
    score = grid[x][y]
    score_ = grid[x_][y_]
    if die_[0]:
        if score_ != score + n_ * die_[0]:
            return None
    else:
        if (score_ - score) % n_ == 0:
            die_copy = list(die_)
            die_copy[0] = (score_ - score) // n_
            die_ = tuple(die_copy)
        else:
            return None

    return (n_, c_, die_, vis_)


def next(state: State) -> list[State]:
    # given state, return list of possible states after 1 move
    ans = []
    dirs = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    for d in dirs:
        new_state = move(d, state)
        if new_state:
            ans.append(new_state)

    return ans


def solve(xs: list[State]) -> list[State]:
    # given list of states, propagate each state until it is final
    ys = [x for x in xs if final(x)]
    zs = [x for x in xs if not final(x)]

    if not zs:
        return xs

    ws = [w for x in zs for w in next(x)]
    return ys + solve(ws)


# answer
# ----------------------------------------------------------------------

final_state = solve([initial])[0]
visited = final_state[3]

tot = sum(grid[i][j] for i in range(m) for j in range(m))
vis = sum(grid[i][j] for i, j in set(visited))
ans = tot - vis


# print
# ----------------------------------------------------------------------


def print_state(state: State):
    n, c, die, vis = state
    output = f"""state:
    n = {n},
    c = {c},
    d = {die},
    v = {vis}.

    """
    print(output)


print_state(final_state)
print(f"answer = {ans}")

# final state:
# n = 32,
# c = (0, 5),
# d = (7, 9, -3, 9, 5, -9),
# v = [(5, 0), (4, 0), (4, 1), (4, 2), (5, 2), (5, 1), (4, 1), (3, 1),
#      (2, 1), (1, 1), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0),
#      (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (5, 3), (5, 4), (5, 5),
#      (4, 5), (3, 5), (3, 4), (3, 3), (2, 3), (1, 3), (1, 4), (1, 5),
#      (0, 5)].

# answer = 1935
