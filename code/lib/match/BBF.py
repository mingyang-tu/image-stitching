import numpy as np
import heapq
from util import MaxHeap


class TreeNode:
    def __init__(self, feature=None, index=None, parent=None, left=None, right=None, split=None):
        self.feature = feature
        self.index = index
        self.parent = parent
        self.left = left
        self.right = right
        self.split = split
        self.dist = float("inf")

    @property
    def brother(self):
        if self.parent is None:
            return None
        else:
            if self.parent.left is self:
                return self.parent.right
            else:
                return self.parent.left

    def __str__(self):
        return f"idx:{str(self.index)}/dist:{str(self.dist)}"

    def __lt__(self, other):
        return self.dist < other.dist


class KDTree:
    def __init__(self):
        self.root = TreeNode()
        self.tree_size = 0

    def __str__(self):
        output = []
        queue = [self.root]
        while queue:
            length = len(queue)
            layer = []
            for _ in range(length):
                curr = queue.pop(0)
                layer.append(str(curr))
                if curr.left is not None:
                    queue.append(curr.left)
                if curr.right is not None:
                    queue.append(curr.right)
            output.append("->".join(layer))
        return "\n".join(output)

    def _choose_feature(self, feature):
        return np.argmax(np.var(feature, axis=0))

    def _split_feature(self, feature):
        idxs = np.argsort(feature)
        mid = idxs.shape[0] // 2
        return (idxs[mid], idxs[:mid], idxs[mid+1:])

    def _min_max(self, feature):
        ROW, COL = feature.shape
        scaled = np.zeros((ROW, COL))
        for i in range(COL):
            M, m = np.max(feature[:, i]), np.min(feature[:, i])
            scaled[:, i] = (feature[:, i] - m) / (M - m)
        return scaled

    def build_tree(self, features):
        def _build_tree(feat, feat_norm, idxs, parent=None):
            if feat.shape[0] == 0:
                return None
            idx_f = self._choose_feature(feat_norm)
            median, lefts, rights = self._split_feature(feat[:, idx_f])
            curr = TreeNode(
                feature=feat[median, :],
                index=idxs[median],
                parent=parent,
                split=idx_f
            )
            curr.left = _build_tree(feat[lefts, :], feat_norm[lefts, :], idxs[lefts], curr)
            curr.right = _build_tree(feat[rights, :], feat_norm[rights, :], idxs[rights], curr)
            return curr

        scaled = self._min_max(features)
        indexs = np.arange(scaled.shape[0])
        self.root = _build_tree(features, scaled, indexs)
        self.tree_size = scaled.shape[0]

    def _trace(self, target, root, heap, seen):
        curr = root
        while curr.left or curr.right:
            if seen[curr.index]:
                return heap
            seen[curr.index] = True
            curr.dist = self._eu_dist(target, curr.feature)
            heapq.heappush(heap, curr)
            if curr.left is None:
                curr = curr.right
            elif curr.right is None:
                curr = curr.left
            else:
                assert curr.feature is not None
                if target[curr.split] < curr.feature[curr.split]:
                    curr = curr.left
                else:
                    curr = curr.right

        if not seen[curr.index]:
            seen[curr.index] = True
            curr.dist = self._eu_dist(target, curr.feature)
            heapq.heappush(heap, curr)
        return heap

    def _eu_dist(self, vec1, vec2):
        return np.sqrt(np.sum(np.square(vec1 - vec2)))

    def _hyper_plane_dist(self, target, split_node):
        idx = split_node.split
        return abs(target[idx] - split_node.feature[idx])

    def knn(self, target, k, max_search):
        neighbor = MaxHeap(k)
        seen = [False] * self.tree_size
        priority_q = self._trace(target, self.root, [], seen)

        last_found = len(priority_q)
        max_search -= last_found
        while max_search > 0 and len(priority_q) > 0:
            curr = heapq.heappop(priority_q)
            neighbor.push(curr)
            if neighbor.peek() > self._hyper_plane_dist(target, curr):
                if curr.brother is not None:
                    priority_q = self._trace(target, curr.brother, priority_q, seen)
            max_search -= len(priority_q) - last_found + 1
            last_found = len(priority_q)

        for _ in range(k):
            if len(priority_q) == 0:
                break
            neighbor.push(heapq.heappop(priority_q))
        return neighbor.heap
