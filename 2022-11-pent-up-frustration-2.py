# Imports
# ----------------------------------------------------------------------
from typing import Optional
from shapely import Polygon
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from time import time

# Types
# ----------------------------------------------------------------------
Point = tuple[float, float]
Pentagon = tuple[Point, Point, Point, Point, Point]
Polyform = list[Pentagon]


# Functions
# ----------------------------------------------------------------------

def plot(polyform: Polyform) -> None:
    plt.figure(figsize=(5, 5))
    for poly in polyform:
        pent = Polygon(poly)
        pts = list(pent.exterior.coords)
        x, y = zip(*pts)
        plt.plot(x, y, linewidth=1, color='black')
    plt.axis('equal')
    plt.show()


alpha = 3 * np.pi / 5
c = np.cos(alpha)
s = np.sin(alpha)
A = np.array([
    [c, -s],
    [s,  c],
])

def pentagon(side: int, x: Point, y: Point) -> Pentagon:
    '''Given points x and y, outputs pentagon with base (x, y).'''
    a: npt.NDArray[np.float64] = np.array(x)
    b: npt.NDArray[np.float64] = np.array(y)

    e = a + A @ (b - a)
    d = e + A @ (a - e)
    c = d + A @ (e - d)

    e = tuple(e)
    d = tuple(d)
    c = tuple(c)
    
    k = (-side) % 5
    ans = [x, y, c, d, e]
    rot = ans[k:] + ans[:k]
    return tuple(rot)


def add(last: int, side: int, polyform: Polyform) -> Optional[Polyform]:
    '''Attach a pentagon to the polyform. Return none if overlap.'''
    if side == last:
        return None
    
    pent = polyform[-1]
    x = pent[(side + 1) % 5]
    y = pent[side]
    new_pent = pentagon(side, x, y)

    # Test if new_pent intersects with other pents. If so, return None
    pent0 = Polygon(new_pent)
    for pent in polyform[:-2]:
        pent1 = Polygon(pent)
        if pent0.intersects(pent1):
            return None
    
    # polyform.append(new_pent)
    return polyform + [new_pent]


def dist(polyform: Polyform) -> float:
    '''Distance between the first and last pentagons of the polyform.'''
    firs = Polygon(polyform[0])
    last = Polygon(polyform[-1])
    return firs.distance(last)


# Define polyforms
# ----------------------------------------------------------------------

# pentagon 0
pent0 = pentagon(0, (0, 0), (1, 0))

# pentagon 1
k = 1
x = pent0[(k+1) % 5]
y = pent0[k]
pent1 = pentagon(k, x, y)

# pentagon 2
k = 4
x = pent1[(k+1) % 5]
y = pent1[k]
pent2 = pentagon(k, x, y)

# polyform
poly = [pent0, pent1, pent2]


# Solution
# ----------------------------------------------------------------------

def best(t: float, k: int) -> Polyform:

    t0 = time()
    best_poly = []
    best_dist = np.inf


    def best_inner(last: int, n: int, polyform: Polyform) -> Polyform:
        nonlocal best_poly
        nonlocal best_dist

        if n == 0 or t < time() - t0:
            return polyform
        if (n + 1) * 1.4 < dist(polyform):
            return polyform

        for side in range(5):
            add_poly = add(last, side, polyform)
            if not add_poly: continue
            p = best_inner(side, n - 1, add_poly)
            d = dist(p)
            if 10**(-6) < d < best_dist - 10**(-6):
                best_poly = p
                best_dist = d
                print(f'd = {best_dist:.7f} (t = {time() - t0:7.4f} sec)')
        
        return best_poly

    return best_inner(-1, k - 3, poly)


p = best(20, 17)
plot(p)

'''
d = 6.7458563 (t =  0.0030 sec)
d = 2.1266270 (t =  0.0060 sec)
d = 1.5388418 (t =  0.0080 sec)
d = 0.3819660 (t =  0.0199 sec)
d = 0.3632713 (t =  0.0449 sec)
d = 0.2245140 (t =  0.0678 sec)
d = 0.1387573 (t = 14.4584 sec)
'''