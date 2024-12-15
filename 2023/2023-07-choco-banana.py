from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from codetiming import Timer
from numpy.typing import NDArray
from pyprune import Backtracking, Choices, Grid  # Library I made for these types of problems
from scipy.ndimage import label

# Types
# ----------------------------------------------------------------------
Nums = NDArray[np.int32]  # Numbers in each cell
Array = NDArray[np.int32]  # General array
Trio = tuple[int, int, int]

WHITE_VALUE = 0
BLACK_VALUE = 1

WHITE = 1 << WHITE_VALUE
BLACK = 1 << BLACK_VALUE
GRAY = WHITE | BLACK
RED = WHITE & BLACK
possible_colors = [RED, WHITE, BLACK, GRAY]

# Input
# ----------------------------------------------------------------------
# fmt: off
z = 0
nums: Nums = np.array([
    [ 6,  6,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  6,  6],
    [ 6,  z,  z,  z,  z,  z,  z,  z,  z,  8, 12,  z,  z,  z,  z,  z,  z,  z,  z,  6],
    [ z,  z,  z, 10, 10,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z, 12, 12,  z,  z,  z],
    [ z,  z,  z, 10,  z,  z, 10, 10,  z,  z,  z,  z, 11, 11,  z,  z,  4,  z,  z,  z],
    [ z,  z,  z,  z,  z,  z, 10,  z,  z,  z,  z,  z,  z, 11,  z,  z,  z,  z,  z,  z],
    [ z, 15,  z,  z,  z,  z,  z,  z,  z,  3,  4,  z,  z,  z,  z,  z,  z,  z,  3,  z],
    [ z,  4,  z,  z,  z,  z,  z,  z,  z,  6,  5,  z,  z,  z,  z,  z,  z,  z, 12,  z],
    [ z,  z,  z,  z,  z,  z,  9,  z,  z,  z,  z,  z,  z,  8,  z,  z,  z,  z,  z,  z],
    [ z,  z,  z, 15,  z,  z,  9,  9,  z,  z,  z,  z,  8,  8,  z,  z,  8,  z,  z,  z],
    [ z,  z,  z,  1,  9,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  1,  7,  z,  z,  z],
    [ 4,  z,  z,  z,  z,  z,  z,  z,  z, 12,  8,  z,  z,  z,  z,  z,  z,  z,  z,  4],
    [ 4,  4,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  4,  4],
], dtype=np.int32)

nums_small: Nums = np.array([
    [z, 6, z, z, z, z, z],
    [z, z, z, z, z, z, 3],
    [z, z, z, 3, z, z, z],
    [z, z, 6, 1, 5, z, z],
    [z, z, z, 4, z, z, z],
    [5, z, z, z, z, z, z],
    [z, z, z, z, z, 4, z],
], dtype=np.int32)
# fmt: on


class ChocoBanana(Backtracking):
    # Precomputations
    # ------------------------------------------------------------------
    def __init__(self, nums: Nums) -> None:
        self.nums = nums
        self.numbered = np.array([(i, j, k) for (i, j), k in np.ndenumerate(nums) if k])
        self.adjacent = self.orthogonally_adjacent(nums)
        self.rectangles = {self.tup(c): self.rectangle_masks(nums, *c) for c in self.numbered}
        self.cm = self.initial(nums, self.numbered, self.rectangles)

    @staticmethod
    def orthogonally_adjacent(nums: Nums) -> Array:
        m, n = nums.shape
        orthogonally_adjacent = []
        for i0, j0 in product(range(m), range(n)):
            for o in range(2):
                di = o
                dj = 1 - o
                i1 = i0 + di
                j1 = j0 + dj
                if not (0 <= i1 < m and 0 <= j1 < n):
                    continue
                if nums[i0, j0] == 0 or nums[i1, j1] == 0:
                    continue
                if nums[i0, j0] == nums[i1, j1]:
                    continue
                orthogonally_adjacent.append((i0, j0, i1, j1))
                orthogonally_adjacent.append((i1, j1, i0, j0))
        return np.array(orthogonally_adjacent, dtype=np.int32)

    def rectangle_masks(self, nums: Nums, i: np.int32, j: np.int32, k: np.int32) -> Array:
        m, n = nums.shape
        masks = []
        rectangle_shapes = [(a, int(k // a)) for a in range(1, k + 1) if k % a == 0]
        for a, b in rectangle_shapes:
            r_min = max(i - a + 1, 0)
            r_max = min(m - a + 1, i + 1)
            c_min = max(j - b + 1, 0)
            c_max = min(n - b + 1, j + 1)
            row_indices = np.arange(r_min, r_max, dtype=np.int32)
            col_indices = np.arange(c_min, c_max, dtype=np.int32)
            black_rectangle = BLACK * np.ones((a, b), dtype=np.int32)
            for r, c in product(row_indices, col_indices):
                mask = np.zeros((m, n), dtype=np.int32)
                mask[r : r + a, c : c + b] = black_rectangle
                digit_mask = mask.astype(bool) * nums
                if np.unique(digit_mask[digit_mask != 0]).size > 1:
                    continue
                white_surrounds = WHITE * self.surroundings_2d(mask.astype(bool))
                mask = np.where(white_surrounds, white_surrounds, mask)
                masks.append(mask[None, :, :])
        return np.concatenate(masks, axis=0) if masks else np.zeros((0, m, n), dtype=np.int32)

    @staticmethod
    def surroundings_2d(mask: NDArray[np.bool]) -> NDArray[np.bool]:
        mask_u = np.pad(mask[:-1, :], ((1, 0), (0, 0)), constant_values=False)
        mask_d = np.pad(mask[1:, :], ((0, 1), (0, 0)), constant_values=False)
        mask_l = np.pad(mask[:, :-1], ((0, 0), (1, 0)), constant_values=False)
        mask_r = np.pad(mask[:, 1:], ((0, 0), (0, 1)), constant_values=False)
        return (mask_u | mask_d | mask_l | mask_r) & ~mask

    def surroundings_3d(self, mask: NDArray[np.bool]) -> NDArray[np.bool]:
        return np.stack([self.surroundings_2d(slice) for slice in mask])

    @staticmethod
    def initial(nums: Nums, numbered: Array, rectangles: dict[Trio, Array]) -> Choices:
        cm = np.where(nums == 1, BLACK, GRAY).astype(np.int32)
        for (i, j, _), mask in zip(numbered, rectangles.values()):
            if mask.shape[0] == 0:
                cm[i, j] &= WHITE
        return cm

    @staticmethod
    def tup(arr: NDArray[np.int32]) -> Trio:
        return tuple(map(int, arr))

    # Expand
    # ------------------------------------------------------------------

    def expand(self, cm: Choices) -> list[Choices]:
        unfilled_numbd = self.unfilled(cm)
        if unfilled_numbd.size:
            # If for some (i, j, k) there are no ways to complete, reject
            expan = [self.expansions(self.rectangles[self.tup(c)], cm, c) for c in unfilled_numbd]
            if any(v.shape[0] == 0 for v in expan):
                return []
            # If all expansions agree on a region, fill just that region and return
            or_masks = [np.bitwise_or.reduce(v, axis=0, keepdims=True) for v in expan]
            or_mask = np.concatenate(or_masks, axis=0)
            or_mask_singleton = or_mask & (or_mask - 1) == 0
            cm_new_stack = np.where(or_mask_singleton, cm & or_mask, cm)
            cm_new = np.bitwise_and.reduce(cm_new_stack, axis=0)
            if not np.array_equal(cm, cm_new):
                return [cm_new]
            # Find cell with fewest possibilities, fill it with white and possible black rectangles
            min_key = lambda arr: arr.shape[0]
            minimal_expansion = min(expan, key=min_key)
            return list(minimal_expansion)
        else:
            # Find an unfilled cell and fill it with black and white
            i, j = np.argwhere(cm == GRAY)[0]
            cm_copies = np.repeat(cm[np.newaxis, ...], 2, axis=0)
            cm_copies[:, i, j] = [WHITE, BLACK]
            return list(cm_copies)

    def unfilled(self, cm: Choices) -> NDArray[np.bool]:
        components = self.connected_components(cm)
        cardinality_array = np.sum(components, axis=(1, 2))
        cardinality_mask = np.sum(components * cardinality_array[:, None, None], axis=0)
        digits_array = np.max(components * self.nums, axis=(1, 2))
        digits_mask = np.sum(components * digits_array[:, None, None], axis=0)
        mask = np.where(cm == BLACK, cardinality_mask < digits_mask, cm == GRAY)
        rows = self.numbered[:, 0]
        cols = self.numbered[:, 1]
        indices = mask[rows, cols]
        return self.numbered[indices]

    @staticmethod
    def connected_components(cm: Choices) -> NDArray[np.bool]:
        color_sets = cm == np.unique(cm)[:, None, None]
        component_sets_list = []
        for color_set in color_sets:
            components, _ = label(color_set)
            nonzero_values = np.unique(components[components != 0])
            component_sets = components == nonzero_values[:, None, None]
            component_sets_list.append(component_sets)
        return np.concatenate(component_sets_list)

    @staticmethod
    def expansions(masks: Array, cm: Choices, cell: Array) -> Array:
        i, j, _ = cell
        if cm[i, j] == GRAY:
            white_mask = np.zeros_like(cm)
            white_mask[i, j] = WHITE
            masks = np.concatenate((np.expand_dims(white_mask, axis=0), masks), axis=0)
        possibilities = np.where(masks, cm & masks, cm)
        valid_indices = np.all(possibilities, axis=(1, 2))
        return possibilities[valid_indices]

    # Prune
    # ------------------------------------------------------------------

    def prune(self, cm: Choices) -> Choices | None:
        rules = [
            self.reject_invalid,
            self.fill_surroundings,
            self.fill_rectangular_closure,
            self.adjacent_regions_different_nums,
            self.orthogonally_adjacent_nums,
            self.black_diagonal,
            self.join_components_with_gray_cell,
            self.unique_path_for_component,
        ]

        cm = np.copy(cm)
        for func in rules:
            cm = func(cm)
            if self.reject(cm):
                return None
        return cm

    @staticmethod
    def rectangle_closure_2d(mask: NDArray[np.bool]) -> NDArray[np.bool]:
        rows, cols = np.where(mask)
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()
        result = np.zeros(mask.shape, dtype=np.bool)
        result[row_min : row_max + 1, col_min : col_max + 1] = True
        return result

    def rectangle_closure_3d(self, mask: NDArray[np.bool]) -> NDArray[np.bool]:
        return np.stack([self.rectangle_closure_2d(slice) for slice in mask])

    def reject_invalid(self, cm: Choices) -> Choices | None:
        components = self.connected_components(cm)
        closures = self.rectangle_closure_3d(components)
        surroundings = self.surroundings_3d(components)
        digits = components * self.nums
        colors = np.max(components * cm, axis=(1, 2))
        cardinality = np.sum(components, axis=(1, 2), dtype=np.int32)
        max_digit = np.max(digits, axis=(1, 2))
        num_digits = np.array([np.unique(digs[digs != 0]).size for digs in digits])
        is_rectangle = np.all(components == closures, axis=(1, 2))
        is_finished = np.all(surroundings * cm != GRAY, axis=(1, 2))

        if np.any((colors == WHITE) & is_finished & is_rectangle):
            return None
        if np.any((colors == BLACK) & is_finished & ~is_rectangle):
            return None
        if np.any((colors != GRAY) & (num_digits > 1)):
            return None
        if np.any((colors != GRAY) & (num_digits == 1) & (cardinality > max_digit)):
            return None
        if np.any((colors != GRAY) & (num_digits == 1) & (cardinality < max_digit) & is_finished):
            return None
        return cm

    def fill_surroundings(self, cm: Choices) -> Choices | None:
        components = self.connected_components(cm)
        surroundings = self.surroundings_3d(components)
        digits = components * self.nums
        colors = np.max(components * cm, axis=(1, 2))
        cardinality = np.sum(components, axis=(1, 2), dtype=np.int32)
        max_digit = np.max(digits, axis=(1, 2))
        num_digits = np.array([np.unique(digs[digs != 0]).size for digs in digits])
        is_finished = np.all(surroundings * cm != GRAY, axis=(1, 2))

        indices = (colors != GRAY) & ~is_finished & (num_digits == 1) & (cardinality == max_digit)
        mask = np.bitwise_or.reduce((colors[:, None, None] * surroundings)[indices], axis=0)
        return np.where(mask, cm & ~mask, cm)

    def fill_rectangular_closure(self, cm: Choices) -> Choices | None:
        components = self.connected_components(cm)
        closures = self.rectangle_closure_3d(components)
        surroundings = self.surroundings_3d(components)
        colors = np.max(components * cm, axis=(1, 2))
        is_rectangle = np.all(components == closures, axis=(1, 2))
        is_finished = np.all(surroundings * cm != GRAY, axis=(1, 2))

        indices = (colors == BLACK) & ~is_finished & ~is_rectangle
        mask = BLACK * np.bitwise_or.reduce(closures[indices], axis=0)
        return np.where(mask, cm & mask, cm)

    def adjacent_regions_different_nums(self, cm: Choices) -> Choices | None:
        components = self.connected_components(cm)
        surroundings = self.surroundings_3d(components)
        digits = components * self.nums
        colors = np.max(components * cm, axis=(1, 2))
        max_digit = np.max(digits, axis=(1, 2))
        num_digits = np.array([np.unique(digs[digs != 0]).size for digs in digits])

        cm = np.copy(cm)
        indices = (colors != GRAY) & num_digits == 1
        for digit, color, surround in zip(
            max_digit[indices], colors[indices], surroundings[indices]
        ):
            surrounding_digits = np.logical_and(surround, self.nums)
            for i, j in np.argwhere(surrounding_digits):
                if self.nums[i, j] != digit:
                    cm[i, j] &= ~color
        return cm

    def orthogonally_adjacent_nums(self, cm: Choices) -> Choices | None:
        cm = np.copy(cm)
        for i0, j0, i1, j1 in self.adjacent:
            for color in [WHITE, BLACK]:
                if cm[i0, j0] == color:
                    cm[i1, j1] &= ~color
        return cm

    def black_diagonal(self, cm: Choices) -> Choices | None:
        m, n = cm.shape
        cm = np.copy(cm)
        for i in range(m - 1):
            for j in range(n - 1):
                for o in range(4):
                    i0, j0 = i + ((o + 0 - 1) % 4 < 2), j + ((o + 0) % 4 > 1)  # main
                    i1, j1 = i + ((o + 1 - 1) % 4 < 2), j + ((o + 1) % 4 > 1)  # diag
                    i2, j2 = i + ((o + 2 - 1) % 4 < 2), j + ((o + 2) % 4 > 1)  # op
                    i3, j3 = i + ((o + 3 - 1) % 4 < 2), j + ((o + 3) % 4 > 1)  # diag
                    if cm[i1, j1] == BLACK and cm[i2, j2] == WHITE and cm[i3, j3] == BLACK:
                        cm[i0, j0] &= WHITE
        return cm

    def join_components_with_gray_cell(self, cm: Choices) -> Choices | None:
        components = self.connected_components(cm)
        surroundings = self.surroundings_3d(components)
        digits = components * self.nums
        colors = np.max(components * cm, axis=(1, 2))
        cardinality = np.sum(components, axis=(1, 2), dtype=np.int32)
        max_digit = np.max(digits, axis=(1, 2))

        cm = np.copy(cm)
        for i, j in np.argwhere(cm == GRAY):
            for color in [BLACK, WHITE]:
                filters = surroundings[:, i, j] & colors == color
                cardinalities = cardinality[filters]
                digit_list = max_digit[filters]
                digit_list = digit_list[digit_list != 0]
                num_unique_digits = np.unique(digit_list).size
                minimal_digit = 10e6 if digit_list.size == 0 else np.min(digit_list)
                if num_unique_digits > 1 or minimal_digit < 1 + np.sum(cardinalities):
                    cm[i, j] &= ~color
        return cm

    def unique_path_for_component(self, cm: Choices) -> Choices | None:
        components = self.connected_components(cm)
        surroundings = self.surroundings_3d(components)
        digits = components * self.nums
        colors = np.max(components * cm, axis=(1, 2))
        cardinality = np.sum(components, axis=(1, 2), dtype=np.int32)
        max_digit = np.max(digits, axis=(1, 2))
        num_digits = np.array([np.unique(digs[digs != 0]).size for digs in digits])

        unique_path = np.sum(np.logical_and(surroundings, cm == GRAY), axis=(1, 2)) == 1
        indices = (colors != GRAY) & (num_digits == 1) & (cardinality < max_digit) & unique_path
        colors_filtered = colors[indices]
        surrounds_filtered = np.logical_and(surroundings[indices], cm == GRAY)
        masks = colors_filtered[:, None, None] * surrounds_filtered
        for mask in masks:
            cm = np.where(mask, cm & mask, cm)
        return cm

    # Answer & plots
    # ------------------------------------------------------------------
    @staticmethod
    def answer(xm: Grid) -> np.int32:
        return np.prod(np.sum(xm == 0, axis=1))

    def plot(self, xm: Grid, title: str = "") -> None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
        annot = np.array(self.nums).astype("str")
        annot[annot == "0"] = ""
        cmap = ["white", "black"]
        ax = sns.heatmap(
            xm,
            annot=annot,
            cbar=False,
            fmt="",
            linewidths=0.1,
            linecolor="black",
            square=True,
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        if title:
            plt.title(title)
        plt.show()


# Solve puzzle
# ----------------------------------------------------------------------

problem = ChocoBanana(nums)
with Timer(initial_text="Solving puzzle..."):
    xm = problem.solution()
ans = problem.answer(xm)
problem.plot(xm)
print(f"{ans = }")
print(xm)
# Solving puzzle...
# Elapsed time: 2.6556 seconds
# ans = np.int64(809321103360)
# [[0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 1]
#  [0 0 1 0 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 1]
#  [0 1 0 0 0 0 0 0 1 0 1 1 1 1 1 1 0 0 0 0]
#  [1 0 1 1 1 1 1 0 0 1 0 0 0 0 0 0 1 1 1 1]
#  [0 0 1 1 1 1 1 0 1 0 1 1 0 0 0 1 0 0 0 0]
#  [0 0 0 0 0 0 0 1 0 0 1 1 0 1 0 1 0 1 1 1]
#  [1 1 0 0 1 1 1 0 1 1 0 0 1 0 1 0 0 0 0 0]
#  [1 1 0 0 1 1 1 0 1 1 0 0 1 0 1 0 1 1 1 1]
#  [0 0 1 0 1 1 1 0 1 1 0 1 0 0 1 0 1 1 1 1]
#  [0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0]
#  [1 1 0 0 1 1 1 1 1 1 0 0 0 1 0 0 1 0 1 1]
#  [1 1 0 0 1 1 1 1 1 1 0 0 0 0 1 1 0 0 1 1]]
