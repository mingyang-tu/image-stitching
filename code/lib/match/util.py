class MaxHeap:
    def __init__(self, max_size):
        self.max_size = max_size
        self.heap_size = 0
        self.heap = []

    def _heapify_down(self, root):
        left = 2 * root + 1
        right = left + 1

        if left < self.heap_size and self.heap[left] > self.heap[root]:
            largest = left
        else:
            largest = root

        if right < self.heap_size and self.heap[right] > self.heap[largest]:
            largest = right

        if largest != root:
            self.heap[largest], self.heap[root] = self.heap[root], self.heap[largest]
            self._heapify_down(largest)

    def _heapify_up(self, root):
        parent = (root - 1) // 2
        if parent >= 0 and self.heap[parent] < self.heap[root]:
            self.heap[parent], self.heap[root] = self.heap[root], self.heap[parent]
            self._heapify_up(parent)

    def pop(self):
        assert self.heap_size > 0, "The heap is empty."
        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
        self.heap_size -= 1
        self._heapify_down(0)
        return self.heap.pop()

    def push(self, node):
        self.heap.append(node)
        self.heap_size += 1
        self._heapify_up(self.heap_size - 1)
        if self.heap_size > self.max_size:
            self.pop()

    def peek(self):
        if self.heap_size < self.max_size:
            return float("inf")
        return self.heap[0].dist