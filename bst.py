from numba.experimental import jitclass
import numpy as np
import numba as nb
from numba import njit

nodeT = nb.deferred_type()

spec = [
    ("data", nb.float32),
    ("left", nb.optional(nodeT)),
    ("right", nb.optional(nodeT)),
]

@jitclass(spec)
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def from_sorted(sorted_elems: np.ndarray) -> Node:
    def build_tree(i: int, j: int) -> Node | None:
        if i >= j:
            return None
        mid = (i + j) // 2
        node = Node(sorted_elems[mid])
        node.left = build_tree(i, mid)
        node.right = build_tree(mid + 1, j)
        return node
    if (out := build_tree(0, len(sorted_elems))) is not None:
        return out
    raise ValueError("Empty list")

# Class members not suppoted yet by numba
@nb.njit
def inorder(node):
    if node.left is not None:
        inorder(node.left)
    print(node.data)
    if node.right is not None:
        inorder(node.right)

@nb.njit
def preorder(node):
    print(node.data)
    if node.left is not None:
        preorder(node.left)
    if node.right is not None:
        preorder(node.right)

@nb.njit
def find(node, value):
    if node.data == value:
        return True
    if value < node.data and node.left is not None:
        return find(node.left, value)
    if value > node.data and node.right is not None:
        return find(node.right, value)
    return False

nodeT.define(Node.class_type.instance_type)


if __name__ == "__main__":
    tree = from_sorted(np.array([1, 2, 3, 4, 5]))
    print("INORDER")
    inorder(tree)
    print("PREORDER")
    preorder(tree)
    print("FIND 3: ", find(tree, 3))
    print("FIND -1: ", find(tree, -1))
    print("FIND 3.4: ", find(tree, 3.4))
