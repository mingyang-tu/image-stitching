from .image_match import ransac


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
                output += f"->{str(curr.index)}({curr.value})({offset[0]:.2f},{offset[1]:.2f})"
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


def swap_pair(pair):
    new_pair = []
    for (s, e) in pair:
        new_pair.append((e, s))
    return new_pair


def prim_mst(graph, pairs, threshold=10):
    N = len(graph)
    vertexs = [ImageNode(i) for i in range(N)]

    root = vertexs[0]
    root.value = float("inf")
    while vertexs:
        curr = extract_max(vertexs)
        if curr.parent:
            if curr.parent.index < curr.index:
                curr.parent.children[curr] = ransac(pairs[curr.parent.index][curr.index])
            else:
                curr.parent.children[curr] = ransac(swap_pair(pairs[curr.index][curr.parent.index]))
        for i in range(N):
            if graph[curr.index][i] >= threshold:
                vert = find_vertex(i, vertexs)
                if vert and graph[curr.index][i] > vert.value:
                    vert.value = graph[curr.index][i]
                    vert.parent = curr
    return root
