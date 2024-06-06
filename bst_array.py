from numba.experimental import jitclass
import numpy as np
import numba as nb
from numba import njit

spec = [
    ("data", nb.float32[:]),
]

@jitclass(spec)
class Tree:
    def __init__(self, data):
        self.data = data

def from_sorted(sorted_elems: np.ndarray) -> Tree:
    level = int(np.ceil(np.log2(len(sorted_elems) + 1)) + 1)
    data = np.full((1 << level) - 1, -1, dtype=np.float32)
    def build(l, r, idx):
        if l > r:
            return
        m = (l + r) // 2
        data[idx] = sorted_elems[m]
        build(l, m - 1, left(idx))
        build(m + 1, r, right(idx))

    build(0, len(sorted_elems) - 1, 0)
    return Tree(data)

@nb.njit
def left(idx):
    return 2*idx + 1

@nb.njit
def right(idx):
    return 2*idx + 2

# Class members not suppoted yet by numba
@nb.njit
def inorder(tree: Tree):

    stack = []
    cur = 0

    while True:
        while tree.data[cur] != -1:
            stack.append(cur)
            cur = left(cur)
        if len(stack) == 0:
            break
        cur = stack.pop()
        print(tree.data[cur])
        cur = right(cur)

@nb.njit
def find(tree: Tree, elem):
    data = tree.data
    cur = 0
    while data[cur] != -1:
        if data[cur] == elem:
            return True
        if data[cur] < elem:
            cur = right(cur)
        else:
            cur = left(cur)
    return False


if __name__ == "__main__":
    tree = from_sorted(np.array([1, 2, 3, 4, 5], dtype=np.float32))
    print("INORDER")
    inorder(tree)
    print("FIND 3: ", find(tree, 3))
    print("FIND -1: ", find(tree, -1))
    print("FIND 3.4: ", find(tree, 3.4))
