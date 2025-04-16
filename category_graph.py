import json
from collections import defaultdict
import heapq

class CategoryGraph:
    def __init__(self):
        # Load problems and extract categories
        with open('problems.json', 'r') as f:
            self.problems = json.load(f)
            self.categories = list(self.problems.keys())
        
        # Define relationships between categories (weights represent learning difficulty)
        self.graph = {
            "Arrays and Hashing": {
                "Two Pointers": 2,
                "Sliding Window": 3,
                "Stack": 3
            },
            "Two Pointers": {
                "Arrays and Hashing": 2,
                "Sliding Window": 2,
                "Linked List": 3
            },
            "Sliding Window": {
                "Two Pointers": 2,
                "Arrays and Hashing": 3,
                "Dynamic Programming 1D": 4
            },
            "Stack": {
                "Arrays and Hashing": 3,
                "Trees": 3,
                "Backtracking": 4
            },
            "Binary Search": {
                "Arrays and Hashing": 3,
                "Two Pointers": 3,
                "Dynamic Programming 1D": 4
            },
            "Linked List": {
                "Two Pointers": 3,
                "Trees": 4
            },
            "Trees": {
                "Stack": 3,
                "Linked List": 4,
                "Tries": 4,
                "Backtracking": 4
            },
            "Tries": {
                "Trees": 4,
                "Heap/Priority Queue": 4
            },
            "Heap/Priority Queue": {
                "Trees": 4,
                "Graphs": 4
            },
            "Backtracking": {
                "Trees": 4,
                "Dynamic Programming 2D": 5
            },
            "Graphs": {
                "Trees": 4,
                "Dynamic Programming 2D": 5
            },
            "Dynamic Programming 1D": {
                "Dynamic Programming 2D": 3,
                "Greedy": 4
            },
            "Dynamic Programming 2D": {
                "Dynamic Programming 1D": 3,
                "Graphs": 5
            },
            "Greedy": {
                "Dynamic Programming 1D": 4,
                "Dynamic Programming 2D": 5
            }
        }

    def get_next_category_via_dijkstra(self, start_categories, target_categories, user_progress):
        def compute_weight(category):
            data = user_progress['categories'][category]
            attempted = data['total_attempted']
            correct = data['correct_solutions']
            time = data['average_time']
            accuracy = correct / attempted if attempted > 0 else 0
            normalized_time = time / 1000  # Assuming 1000s is a reasonable max
            return (1 - accuracy) * 0.7 + normalized_time * 0.3

        visited = set()
        heap = []

        for start in start_categories:
            heapq.heappush(heap, (0, [start]))

        while heap:
            cost, path = heapq.heappop(heap)
            current = path[-1]

            if current in target_categories:
                if len(path) >= 2:
                    return path[1]  # Return next category in path
                else:
                    return current

            if current in visited:
                continue
            visited.add(current)

            neighbors = self.graph.get(current, {})
            for neighbor, base_weight in neighbors.items():
                if neighbor not in visited:
                    dynamic_weight = base_weight * compute_weight(neighbor)
                    heapq.heappush(heap, (cost + dynamic_weight, path + [neighbor]))

        return None  # fallback if no path found
