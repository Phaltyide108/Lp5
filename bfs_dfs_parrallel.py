from multiprocessing import Pool

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        self.graph.setdefault(u, []).append(v)
        self.graph.setdefault(v, []).append(u)

def bfs_parallel(graph, start):
    visited, queue = set(), [start]
    visited.add(start)
    
    while queue:
        current_node = queue.pop(0)
        print(current_node, end=' ')
        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

def dfs_parallel(graph, start):
    visited = set()
    def dfs_helper(node):
        print(node, end=' ')
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs_helper(neighbor)
    dfs_helper(start)

if __name__ == "__main__":
    graph = Graph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(1, 4)
    graph.add_edge(2, 5)

    print("BFS:")
    with Pool() as p:
        p.apply(bfs_parallel, args=(graph.graph, 0))
    print("\nDFS:")
    with Pool() as p:
        p.apply(dfs_parallel, args=(graph.graph, 0))
