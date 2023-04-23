import numpy as np


def image_match(lengths, offsets):
    matching_tree = prim_mst(lengths, offsets)

    print("\nRelationship between images:")
    print(matching_tree)

    return matching_tree


def prim_mst(lengths, offsets):
    N = len(lengths)

    vertexs = [ImageNode(i) for i in range(N)]

    root = vertexs[0]
    root.value = float("inf")
    while vertexs:
        curr = extract_max(vertexs)
        if curr.parent:
            curr.parent.children[curr] = offsets[curr.parent.index][curr.index]
        for i in range(N):
            if lengths[curr.index][i] > 0:
                vert = find_vertex(i, vertexs)
                if vert and lengths[curr.index][i] > vert.value:
                    vert.value = lengths[curr.index][i]
                    vert.parent = curr
    return root


class ImageNode:
    def __init__(self, index, value=0., parent=None):
        self.index = index
        self.value = value
        self.parent = parent
        self.children = dict()

    def __str__(self):
        output = str(self.index) + f"({self.value})" + "(root)"
        queue = [(child, offset) for child, offset in self.children.items()]
        while queue:
            output += "\n"
            l = len(queue)
            for i in range(l):
                curr, offset = queue.pop(0)
                output += f"->{str(curr.index)}({curr.value})({offset[0]},{offset[1]})"
                for child, offset in curr.children.items():
                    queue.append((child, offset))
        return output

    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other):
        return self.index == other.index

    def __lt__(self, other):
        return self.value < other.value


def extract_max(array):
    for i in range(1, len(array)):
        if array[i] > array[0]:
            array[0], array[i] = array[i], array[0]
    return array.pop(0)


def find_vertex(index, array):
    for item in array:
        if index == item.index:
            return item
    return None

