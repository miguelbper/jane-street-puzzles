from functools import reduce
from itertools import product
from operator import mul

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from codetiming import Timer
from numpy.typing import NDArray
from pyprune import Backtracking
from scipy.ndimage import label, sum_labels

rows = np.array([810, 585, 415, 92, 67, 136, 8, 225, 567])
cols = np.array([28, 552, 64, 15, 86, 1304, 170, 81, 810])
IntArray = NDArray[np.int32]
BoolArray = NDArray[np.bool]


class Hooks4(Backtracking):
    """Notation:

    cm = [xm | os]: (n, n + 1) matrix int
    bm: (n, n) matrix int
    os: (n, 1) matrix / (n,) vector int

    bm -(log)-> B: (n, n) matrix bool
    os -(log)-> O: (n, 1) matrix / (n,) vector int

    hooks: (n, n) matrix int, with numbers of each hook
    X = B * hooks: (n, n) matrix int
    """

    # Init and utils
    # ------------------------------------------------------------------
    def __init__(self, rows: IntArray, cols: IntArray) -> None:
        n = rows.shape[0]
        self.rows = rows
        self.cols = cols
        self.masks = np.array(list(product(range(2), repeat=n)))

        xm = sum(1 << b for b in range(2)) * np.ones((n, n), dtype=np.int32)
        os = sum(1 << b for b in range(4)) * np.ones((n,), dtype=np.int32)
        os[n - 1] = 1 << 0
        self.cm = self.join(xm, os)

    @staticmethod
    def join(xm: IntArray, os: IntArray) -> IntArray:
        n, _ = xm.shape
        return np.concatenate([xm, os.reshape(n, 1)], axis=1)

    @staticmethod
    def split(cm: IntArray) -> tuple[IntArray, IntArray]:
        n, _ = cm.shape
        xm = cm[:, :n]
        os = cm[:, n]
        return xm, os

    @staticmethod
    def singletons(cm: IntArray, dtype: type = np.int32, unk_value: int = 0) -> IntArray:
        power_of_two = (cm > 0) & ((cm & (cm - 1)) == 0)
        ans = unk_value * np.ones_like(cm)
        ans[power_of_two] = np.log2(cm[power_of_two])
        return ans.astype(dtype)

    def get_hooks(self, os: IntArray) -> IntArray:
        n = os.shape[0]
        hooks = np.zeros((n, n), dtype=np.int32)
        I = np.arange(0, n)[:, np.newaxis]
        J = np.arange(0, n)[np.newaxis, :]

        i0, j0 = 0, 0

        for k, o in enumerate(self.singletons(os, unk_value=-1)):
            if o == -1:
                break

            i1 = i0 + n - k
            j1 = j0 + n - k

            row_cond = o == 0 or o == 3
            col_cond = o == 0 or o == 1
            row = i0 if row_cond else i1 - 1
            col = j0 if col_cond else j1 - 1
            in_row = (I == row) & (j0 <= J) & (J < j1)  # noqa: SIM300
            in_col = (J == col) & (i0 <= I) & (I < i1)  # noqa: SIM300
            mask = in_row | in_col
            digit = n - k
            hooks += digit * mask

            i0 += row_cond
            j0 += col_cond

        return hooks

    @staticmethod
    def score_1d(X: IntArray) -> np.int32:
        n = X.shape[-1]
        I = np.eye(n, dtype=bool)
        M = X.astype(np.bool)
        starts = np.argwhere(M & (~np.roll(M, 1) | I[0]))
        ends = np.argwhere(M & (~np.roll(M, -1) | I[n - 1])) + 1
        cum_prod = np.cumulative_prod(np.where(X, X, 1), include_initial=True)
        products = cum_prod[ends] // cum_prod[starts]
        return np.sum(products)

    def score(self, X: IntArray) -> IntArray:
        return np.vectorize(self.score_1d, signature="(n)->()")(X)

    # Expand
    # ------------------------------------------------------------------
    def expand(self, cm: IntArray) -> list[IntArray]:
        isnt_singleton = (cm & (cm - 1)) != 0
        indices = np.argwhere(isnt_singleton)
        sorted_indices = indices[np.lexsort((indices[:, 0], -indices[:, 1]))]
        i, j = tuple(sorted_indices[0])

        powers_of_two = 1 << np.arange(4)
        powers_present = powers_of_two[cm[i, j] & powers_of_two > 0]
        cm_copies = np.repeat(cm[np.newaxis, ...], len(powers_present), axis=0)
        cm_copies[:, i, j] = powers_present
        return list(cm_copies)

    # Prune
    # ------------------------------------------------------------------
    def prune(self, cm: IntArray) -> IntArray | None:
        funcs = [
            self.connectedness,
            self.num_digits,
            self.scores,
        ]
        for func in funcs:
            cm = func(cm)
            if cm is None:
                return None
        return cm

    def connectedness(self, cm: IntArray) -> IntArray | None:
        xm, _ = self.split(cm)
        xm = self.singletons(xm, dtype=np.bool, unk_value=1)
        return cm if label(xm)[1] <= 1 else None

    def num_digits(self, cm: IntArray) -> IntArray | None:
        n, _ = cm.shape
        xm, os = self.split(cm)
        hooks = self.get_hooks(os)  # (n, n) [i, j]
        hook_indices = n - hooks  # (n, n) [i, j]
        ds = np.arange(2)  # (2,) [d]
        ks = np.arange(n)  # (n,) [k]
        hook_indices_levels = hook_indices == ks.reshape(n, 1, 1)  # (n, n, n) [k, i, j]
        xm_levels = np.where(hook_indices_levels, xm, 0)  # (n, n, n) [k, i, j]
        xm_is_ds = xm_levels == (1 << ds).reshape(2, 1, 1, 1)  # (2, n, n, n) [d, k, i, j]
        num_ds = np.sum(xm_is_ds, axis=(2, 3))  # (2, n) [d, k]
        max_ds = n + ds.reshape(2, 1) - ks - 1  # (2, n) [d, k]
        if np.any(num_ds > max_ds):
            return None

        num_is_max = num_ds == max_ds  # (2, n) [d, k]
        mask = (xm_levels == 3) & num_is_max.reshape(2, n, 1, 1)  # (2, n, n, n) [d, k, i, j]
        xm_levels_masked = np.where(
            mask, 1 << (1 - ds.reshape(2, 1, 1, 1)), xm_levels
        )  # (2, n, n, n) [d, k, i, j] # noqa: E501
        update = np.sum(xm_levels_masked, axis=1)  # (2, n, n) [d, i, j]
        xm_new = np.copy(xm)  # (n, n) [i, j]
        for upd in update:
            xm_new = np.where(upd, xm_new & upd, xm_new)  # (n, n) [i, j]

        cm_new = self.join(xm_new, os)
        return cm_new

    def scores(self, cm: IntArray) -> IntArray | None:
        xm, os = self.split(cm)
        hooks = self.get_hooks(os)  # (n, n) [i, j]
        xm_new = np.copy(xm)
        for transpose in range(2):
            nums = cols if transpose else rows
            xm_t = xm.T if transpose else xm
            hooks_t = hooks.T if transpose else hooks
            update = self.scores_update(xm_t, hooks_t, nums)
            if update is None:
                return None
            update_t = update.T if transpose else update
            xm_new = np.where(update_t, xm_new & update_t, xm_new)
        cm_new = self.join(xm_new, os)
        return cm_new

    def scores_update(self, xm: IntArray, hooks: IntArray, nums: IntArray) -> IntArray | None:
        n, _ = xm.shape
        ds = np.arange(2).reshape(2, 1, 1, 1)  # (2, 1, 1, 1) [d, ...]
        xm_is_ds = xm == 1 << ds  # (2, 1, n, n) [d, k, i, j]
        mask_is_ds = self.masks.reshape(2**n, 1, n) == ds  # (2, 2**n, 1, n) [d, k, i, j]
        compatible = ~xm_is_ds | mask_is_ds  # (2, 2**n, n, n) [d, k, i, j]
        compatible_masks = np.all(compatible, axis=(0, 3))  # (2**n, n) [k, i]
        unknown_row = np.any(hooks == 0, axis=1)  # (n,) [i]
        hooks_masked = hooks * self.masks.reshape(2**n, 1, n)  # (2**n, n, n) [k, i, j]
        score = self.score(hooks_masked)  # (2**n, n) [k, i]
        valid_masks = compatible_masks & (unknown_row | (score == nums))  # (2**n, n) [k, i]

        if not np.all(np.any(valid_masks, axis=0)):
            return None

        update = np.zeros_like(xm, dtype=np.int32)  # (n, n) [i, j]
        for i in range(n):
            row_masks = self.masks[valid_masks[:, i], :]  # (?, n) [k, j]
            masks_agree = np.all(row_masks[0] == row_masks, axis=0)
            update[i] = (1 << row_masks[0]) * masks_agree

        return update

    # Print solution
    # ------------------------------------------------------------------
    @staticmethod
    def answer(sol: IntArray) -> int:
        mat = np.where(sol == 0, 1, 0)
        labels, k = label(mat)
        area = sum_labels(mat, labels, index=range(1, k + 1))
        return int(reduce(mul, area))

    def plot(self, X: IntArray, hooks: IntArray) -> None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
        annot = np.array(X).astype("str")
        annot[annot == "0"] = ""

        color_cycle = ["green", "orange", "red", "purple"]
        base_colors = ["white", "blue"] + color_cycle + color_cycle
        reduced_alpha_colors = [(mcolors.to_rgba(c, alpha=0.6)) for c in base_colors]
        cmap = mcolors.ListedColormap(reduced_alpha_colors)

        ax = sns.heatmap(
            hooks,
            annot=annot,
            cbar=False,
            fmt="",
            linewidths=0.1,
            linecolor="black",
            square=True,
            cmap=cmap,
            vmin=0,
            vmax=9,
        )
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        plt.show()


# State and solve the problem
# ----------------------------------------------------------------------
problem = Hooks4(rows, cols)
with Timer(initial_text="Solving problem..."):
    sol = problem.solution()

B, O = problem.split(sol)
hooks = problem.get_hooks(1 << O)
X = B * hooks
ans = problem.answer(X)

print(f"{ans = }")
print(f"{X}")
problem.plot(X, hooks)
# Solving problem...
# Elapsed time: 9.2542 seconds
# ans = 6000
# [[9 9 0 0 0 0 9 9 9]
#  [0 8 8 0 8 8 8 0 9]
#  [0 0 7 7 7 0 0 8 9]
#  [6 6 0 0 0 0 7 8 0]
#  [0 5 5 0 0 6 7 0 0]
#  [4 4 0 4 5 6 0 0 0]
#  [0 2 0 0 0 6 0 0 0]
#  [3 2 1 0 5 6 7 0 9]
#  [3 0 3 4 5 0 7 8 9]]
