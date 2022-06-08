import numpy as np
from itertools import product

# the following is the solution to this puzzle
# this program just checks that the solution is correct

board = np.array([[4, 3,6, 5,3,7,4,9,6,5],
                  [8,10,2, 4,1,1,2,3,8,2],
                  [9, 2,3, 2,1,2,5,1,2,4],
                  [5, 7,2, 1,2,6,3,1,1,3],
                  [6, 3,1, 1,3,2,1,4,2,7],
                  [1, 1,4, 5,1,1,1,3,5,6],
                  [3, 1,2, 3,2,4,2,1,2,3],
                  [4, 2,1, 1,1,1,3,1,4,9],
                  [5, 8,3, 4,2,1,6,2,3,8],
                  [7, 6,9,10,5,3,4,7,2,5]])

solution = sum(np.prod(board, axis = 1))
print('solution = sum of products = {}'.format(solution))
# solution:
# sum of products = 24405360

# define regions and parameters of the board
region_map = np.array([[ 0, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                       [ 0, 0, 1, 1, 1, 2, 3, 3, 2, 2],
                       [ 0, 0, 4, 4, 5, 5, 6, 3, 7, 7],
                       [ 0, 0, 8, 4, 9, 6, 6, 6, 7, 7],
                       [ 0, 8, 8, 9, 9,10,11, 6, 6, 7],
                       [ 0,12, 8,16,22,10,21,21, 7, 7],
                       [ 0,15,13,16,16,16,21,14,14,14],
                       [15,15,13,17,16,18,19,20,20,19],
                       [15,15,15,17,17,19,19,20,20,19],
                       [15,15,15,15,17,17,19,19,19,19]])
(x, y) = region_map.shape
n_regions = np.amax(region_map) + 1
regions = [[(j, k) for (j, k) in product(range(x), range(y)) if region_map[j,k] == i] for i in range(n_regions)]

# check that every region is filled by the numbers 1,...,N
for i in range(len(regions)):
    numbers = [board[j, k] for (j,k) in regions[i]]
    l = len(numbers)
    correct = list(range(1, l+1)) == sorted(numbers)
    print('It is {} that region {:2d} is 1,...,N'.format(correct, i))

# check that every n is at distance n to the closest other n
for (i,j) in product(range(x), range(y)):
    n = board[i,j]
    other_ns = [(k,l) for (k,l) in product(range(x), range(y)) if n == board[k,l] and (i,j) != (k,l)]
    distances = [abs(i-k) + abs(j-l) for (k,l) in other_ns]
    correct = n == min(distances)
    print('For (i,j) = {}, n = {:2d} and min_distance = {:2d}. It is {} that n = min_distance'.format((i,j), n, min(distances), correct))