import numpy as np


def image_match(pairs):
    matching_tree = prim_mst(pairs)

    print("\nRelationship between images:")
    print(matching_tree)

    return matching_tree


def prim_mst(pairs, threshold=10):
    N = len(pairs)

    graph = [[0 for j in range(N)] for i in range(N)]
    for i in range(N):
        for j in range(N):
            if i > j:
                graph[i][j] = len(pairs[j][i])
            else:
                graph[i][j] = len(pairs[i][j])

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


def ransac(pair, k=50, threshold=5):
    max_inlier = 0
    best_shift = (0, 0)
    for _ in range(k):
        select = np.random.randint(0, len(pair))
        (x1, y1), (x2, y2) = pair[select]
        dx, dy = x1 - x2, y1 - y2

        inlier = 0
        dx_sum, dy_sum = 0, 0
        for (x1, y1), (x2, y2) in pair:
            _dx, _dy = x1 - x2, y1 - y2
            if ((_dx - dx) ** 2 + (_dy - dy) ** 2) ** (1/2) <= threshold:
                inlier += 1
                dx_sum += _dx
                dy_sum += _dy
        if inlier > max_inlier:
            max_inlier = inlier
            best_shift = (round(dx_sum / inlier), round(dy_sum / inlier))
    print(f"Ratio of inliers: {max_inlier}/{len(pair)} = {max_inlier / len(pair) * 100:.2f}%")
    return best_shift


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
