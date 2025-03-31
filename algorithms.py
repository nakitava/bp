import itertools
import numpy as np

class Item:
    def __init__(self, weight, value):
        self.weight = int(weight)
        self.value = int(value)
        self.ratio = value / weight if weight != 0 else 0

def generate_items(n, max_weight=30, max_value=100):
    """Генерация предметов с разумными параметрами"""
    weights = np.random.randint(1, max_weight, n)
    values = np.random.randint(1, max_value, n)
    return [Item(w, v) for w, v in zip(weights, values)]

def brute_force(items, capacity):
    max_value = 0
    best_subset = []
    for r in range(1, len(items) + 1):
        for subset in itertools.combinations(items, r):
            total_weight = sum(item.weight for item in subset)
            total_value = sum(item.value for item in subset)
            if total_weight <= capacity and total_value > max_value:
                max_value = total_value
                best_subset = subset
    return best_subset, max_value

def greedy_algorithm(items, capacity):
    sorted_items = sorted(items, key=lambda x: x.ratio, reverse=True)
    total_weight = 0
    total_value = 0
    selected_items = []
    for item in sorted_items:
        if total_weight + item.weight <= capacity:
            selected_items.append(item)
            total_weight += item.weight
            total_value += item.value
    return selected_items, total_value

def dynamic_programming(items, capacity):
    n = len(items)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if items[i-1].weight <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w - items[i-1].weight] + items[i-1].value)
            else:
                dp[i][w] = dp[i-1][w]
    w = capacity
    selected_items = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected_items.append(items[i-1])
            w -= items[i-1].weight
    return selected_items, dp[n][capacity]

def branch_and_bound(items, capacity):
    items = sorted(items, key=lambda x: x.ratio, reverse=True)
    best_value = 0
    best_subset = []
    def backtrack(i, current_weight, current_value, selected):
        nonlocal best_value, best_subset
        if current_weight > capacity:
            return
        if current_value > best_value:
            best_value = current_value
            best_subset = selected.copy()
        if i == len(items):
            return
        selected.append(items[i])
        backtrack(i + 1, current_weight + items[i].weight, current_value + items[i].value, selected)
        selected.pop()
        backtrack(i + 1, current_weight, current_value, selected)
    backtrack(0, 0, 0, [])
    return best_subset, best_value