import numpy as np
from collections import deque
import time
import threading
import multiprocessing
import os

# Function to add edges to the graph
def add_edge(adj, u, v):
    adj[u].append(v)
    adj[v].append(u)

# Function for parallel Breadth First Search
def parallel_bfs(graph, start):
    num_threads = multiprocessing.cpu_count() # Get the number of available CPU cores
    visited = [False] * len(graph)
    visited[start] = True
    queue = deque([start])
    while queue:
        current_node = queue.popleft()
        print("Visited node:", current_node)
        # Parallelize the exploration of neighbors
        with multiprocessing.Pool(processes=num_threads) as pool:
            result = pool.starmap(bfs_helper, [(graph, current_node, neighbor, visited) for neighbor in graph[current_node] if not visited[neighbor]])
            visited = [visited[node] or result[i] for i, node in enumerate(graph[current_node])]

# Helper function for parallel BFS
def bfs_helper(graph, current_node, neighbor, visited):
    print("Thread", os.getpid(), "is exploring neighbor", neighbor)
    visited[neighbor] = True
    return visited[neighbor]

# Function for parallel Depth First Search
def parallel_dfs(graph, start):
    num_threads = multiprocessing.cpu_count() # Get the number of available CPU cores
    visited = [False] * len(graph)
    visited[start] = True
    stack = [start]
    while stack:
        current_node = stack.pop()
        print("Visited node:", current_node)
        # Parallelize the exploration of neighbors
        with multiprocessing.Pool(processes=num_threads) as pool:
            result = pool.starmap(dfs_helper, [(graph, current_node, neighbor, visited) for neighbor in graph[current_node] if not visited[neighbor]])
            visited = [visited[node] or result[i] for i, node in enumerate(graph[current_node])]

# Helper function for parallel DFS
def dfs_helper(graph, current_node, neighbor, visited):
    print("Thread", os.getpid(), "is exploring neighbor", neighbor)
    visited[neighbor] = True
    return visited[neighbor]

# Test the algorithms with a sample graph
if __name__ == "__main__":
    # Sample undirected graph
    graph = [[] for _ in range(6)]
    add_edge(graph, 0, 1)
    add_edge(graph, 0, 2)
    add_edge(graph, 1, 3)
    add_edge(graph, 1, 4)
    add_edge(graph, 2, 5)

    print("Parallel Breadth First Search:")
    start_time = time.time()
    parallel_bfs(graph, 0)
    print("Time taken:", time.time() - start_time)

    print("\nParallel Depth First Search:")
    start_time = time.time()
    parallel_dfs(graph, 0)
    print("Time taken:", time.time() - start_time)
