import multiprocessing
import random

# Function to perform parallel reduction for finding min, max, sum, and average
def parallel_reduction(arr):
    num_processes = multiprocessing.cpu_count()
    chunk_size = len(arr) // num_processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Divide the array into chunks and calculate min, max, and sum in parallel
    results = pool.map(partial_reduction, [arr[i:i+chunk_size] for i in range(0, len(arr), chunk_size)])
    pool.close()
    pool.join()

    # Combine the results from each chunk
    min_val = min(r[0] for r in results)
    max_val = max(r[1] for r in results)
    sum_val = sum(r[2] for r in results)
    avg_val = sum_val / len(arr)

    return min_val, max_val, sum_val, avg_val

# Function to perform reduction on a chunk of the array
def partial_reduction(arr):
    min_val = min(arr)
    max_val = max(arr)
    sum_val = sum(arr)

    return min_val, max_val, sum_val

# Generate a random array for testing
arr = [random.randint(1, 1000) for _ in range(10000)]

# Perform parallel reduction to find min, max, sum, and average
min_val, max_val, sum_val, avg_val = parallel_reduction(arr)
print(f"Min: {min_val}, Max: {max_val}, Sum: {sum_val}, Average: {avg_val}")
