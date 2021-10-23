import numpy as np

import warnings

store_set = []
store_counts = []
store = []


class Node:
    def __init__(self, data):
        self.data = data
        self.child = {}
        self.nodes = []
        self.parent = None
        self.node_parent = self

    def add_child(self, child):
        child.parent = self
        child.node_parent = self.node_parent

        self.child[child.data] = 1
        self.nodes.append(child)

    def print_child(self):
        print(f"parent : {self.parent.data}") if self.parent != None else print(
            f"parent : {self.data}"
        )
        print(f"data : {self.data}")
        print(f"child : {self.child}")
        if self.nodes:
            for node in self.nodes:
                node.print_child()

    def check_add_child(self, tree):
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=DeprecationWarning)

        position = np.where(tree.data == np.array(list(self.child.keys())))[0]
        if len(position) > 0:
            position_col = list(self.child.keys())[position[0]]

            self.child[position_col] += 1

            node = self.nodes[position[0]]
            for t_node in tree.nodes:
                node.check_add_child(t_node)

        else:
            tree.parent = self
            tree.node_parent = self.node_parent

            self.child[tree.data] = 1
            self.nodes.append(tree)

    def find_set(self, endswith):
        global store_counts
        global store_set
        global store

        if endswith in list(self.child.keys()):
            store = []
            counts = self.child[endswith]

            self.back_node_parent(self)

            store_set.append(store)
            store_counts.append(counts)

        for node in self.nodes:
            node.find_set(endswith)

        if self.data == None:
            set_map = {}
            for st_set, count in zip(store_set, store_counts):
                st_set.reverse()

                tup_set = tuple(st_set)
                set_map[tup_set] = count

            store_set = []
            store_counts = []
            store = []

            return set_map

    def back_node_parent(self, node):
        global store

        if node.data == None:
            return 0
        else:
            store.append(node.data)
            node.back_node_parent(node.node_parent)


def make_tree(lst):
    trees = [Node(l) for l in lst]

    node = None
    for i in range(len(lst) - 1, -1, -1):
        node = trees[i] if i == 0 else trees[i - 1].add_child(trees[i])

    return node
