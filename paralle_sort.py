import multiprocessing
import random
import time

def parallel_bubble_sort(arr):
    n = len(arr)

    for i in range(n):
        swapped = False

        with multiprocessing.Pool() as pool:
            results = pool.starmap_async(compare_swap, [(arr, j) for j in range(n - i - 1)])
            pool.close()
            pool.join()

        for result in results.get():
            if result:
                swapped = True

        if not swapped:
            break

    return arr

def compare_swap(arr, j):
    if arr[j] > arr[j + 1]:
        arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return True
    return False

def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    left = merge_sort(left)
    right = merge_sort(right)

    return merge(left, right)

def merge(left, right):
    merged = []
    left_idx, right_idx = 0, 0

    while left_idx < len(left) and right_idx < len(right):
        if left[left_idx] < right[right_idx]:
            merged.append(left[left_idx])
            left_idx += 1
        else:
            merged.append(right[right_idx])
            right_idx += 1

    merged.extend(left[left_idx:])
    merged.extend(right[right_idx:])

    return merged

# Sequential bubble sort for performance comparison
def sequential_bubble_sort(arr):
    n = len(arr)

    for i in range(n):
        swapped = False

        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        if not swapped:
            break

    return arr

# Generate random array for sorting
arr = [random.randint(1, 100) for _ in range(100)]

# Measure time taken by sequential bubble sort
start_time = time.time()
sequential_bubble_sort(arr.copy())
sequential_time = time.time() - start_time
print(f"Sequential Bubble Sort Time: {sequential_time} seconds")

# Measure time taken by parallel bubble sort
start_time = time.time()
parallel_bubble_sort(arr.copy())
parallel_time = time.time() - start_time
print(f"Parallel Bubble Sort Time: {parallel_time} seconds")

# Measure time taken by sequential merge sort
start_time = time.time()
merge_sort(arr.copy())
sequential_merge_time = time.time() - start_time
print(f"Sequential Merge Sort Time: {sequential_merge_time} seconds")

# Measure time taken by parallel merge sort
start_time = time.time()
merge_sort(arr.copy())
parallel_merge_time = time.time() - start_time
print(f"Parallel Merge Sort Time: {parallel_merge_time} seconds")
