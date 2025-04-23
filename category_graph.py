import json
from collections import defaultdict
import heapq
import logging

class CategoryGraph:
    def __init__(self):
        # Setup logger
        self.logger = self._setup_logger()
        
        # Load problems and extract categories
        with open('problems.json', 'r') as f:
            self.problems = json.load(f)
            self.categories = list(self.problems.keys())
        
        self.logger.info("Loaded %d categories of problems", len(self.categories))
        
        # Create a mapping from problem name to category for faster lookups
        self.problem_to_category = {}
        for category_name, problems_list in self.problems.items():
            for problem in problems_list:
                self.problem_to_category[problem['name']] = problem['category']
        
        # Define relationships between categories (weights represent learning difficulty)
        self.graph = {
            "Arrays and Hashing": {
                "Two Pointers": 1.5,
                "Sliding Window": 2.0,
                "Stack": 2.5,
                "Binary Search": 2.5,
                "Greedy": 3.0
            },
            "Two Pointers": {
                "Arrays and Hashing": 1.5,
                "Sliding Window": 1.5,
                "Linked List": 2.5,
                "Binary Search": 2.0
            },
            "Sliding Window": {
                "Two Pointers": 1.5,
                "Dynamic Programming 1D": 3.0,
                "Greedy": 2.5
            },
            "Stack": {
                "Trees": 2.0,
                "Backtracking": 3.0,
                "Graphs": 3.5
            },
            "Binary Search": {
                "Dynamic Programming 1D": 3.0,
                "Greedy": 2.5
            },
            "Linked List": {
                "Trees": 3.0
            },
            "Trees": {
                "Tries": 2.0,
                "Heap / Priority Queue": 2.5,
                "Backtracking": 3.0,
                "Graphs": 2.5
            },
            "Tries": {
                "Heap / Priority Queue": 3.0
            },
            "Heap / Priority Queue": {
                "Graphs": 2.5,
                "Greedy": 2.0
            },
            "Backtracking": {
                "Dynamic Programming 2D": 3.0,
                "Graphs": 2.5
            },
            "Graphs": {
                "Dynamic Programming 2D": 3.0
            },
            "Dynamic Programming 1D": {
                "Dynamic Programming 2D": 2.0,
                "Greedy": 2.5
            },
            "Dynamic Programming 2D": {
            },
            "Greedy": {
                "Dynamic Programming 2D": 3.5
            }
        }

    def _setup_logger(self):
        """Setup logger"""
        # Create logger
        logger = logging.getLogger('category_graph')
        logger.setLevel(logging.DEBUG)
        
        # Create file handler
        file_handler = logging.FileHandler('recommendation_log.txt', mode='w')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        return logger

    def get_problem_category(self, problem_name):
        """Get the category a problem belongs to"""
        return self.problem_to_category.get(problem_name)

    def get_next_category_via_dijkstra(self, start_categories, target_categories, user_progress):
        def compute_weight(category):
            data = user_progress['categories'][category]
            attempted = data['total_attempted']
            correct = data['correct_solutions']
            # If average_time field doesn't exist, use default value 0
            time = data.get('average_time', 0)
            
            # Calculate accuracy
            accuracy = correct / attempted if attempted > 0 else 0
            normalized_time = time / 1000  # Assuming 1000s is a reasonable max
            
            # Calculate average score for all attempted problems in this category
            avg_problem_score = 0.0
            problem_scores_count = 0
            
            # Log debug info
            self.logger.info(f"Computing weight for category '{category}':")
            self.logger.info(f"  - Accuracy: {accuracy:.2f}")
            self.logger.info(f"  - Normalized time: {normalized_time:.2f}")
            
            # Loop through all problems, find those belonging to this category and calculate average score
            for problem_name, problem_data in user_progress.get('problems', {}).items():
                # Check if problem has solutions
                if problem_data.get('solutions', []):
                    # Get score from the latest solution
                    latest_solution = problem_data['solutions'][-1]
                    score = latest_solution.get('score', 0.0)
                    
                    # Get problem category
                    problem_category = self.get_problem_category(problem_name)
                    
                    # If problem belongs to current category, include in calculation
                    if problem_category == category:
                        self.logger.info(f"  - Problem '{problem_name}' score: {score:.2f}")
                        avg_problem_score += score
                        problem_scores_count += 1
            
            # Calculate average score
            if problem_scores_count > 0:
                avg_problem_score /= problem_scores_count
                self.logger.info(f"  - Category average score: {avg_problem_score:.2f} (based on {problem_scores_count} problems)")
            else:
                self.logger.info(f"  - No solved problems in this category")
                avg_problem_score = 0.5  # For categories with no solved problems, use medium score
            
            # New weight formula, considering accuracy, average problem score, and time
            weight = (1 - accuracy) * 0.4 + (1 - avg_problem_score) * 0.3 + normalized_time * 0.3
            self.logger.info(f"  - Final weight: {weight:.2f}")
            return weight

        visited = set()
        # Priority queue with distances from start categories
        distances = {category: float('infinity') for category in self.graph}
        # Initialize distances for start categories
        for start in start_categories:
            distances[start] = 0
        
        # Priority queue for Dijkstra
        queue = [(0, start) for start in start_categories]
        heapq.heapify(queue)
        
        # Previous nodes to reconstruct path
        previous = {category: None for category in self.graph}
        
        self.logger.info("Starting Dijkstra algorithm to find optimal path")
        self.logger.info(f"Start categories: {start_categories}")
        self.logger.info(f"Target categories: {target_categories}")
        
        # Dijkstra algorithm
        while queue:
            # Get node with smallest distance
            current_distance, current_node = heapq.heappop(queue)
            
            # If we've reached a target, we're done
            if current_node in target_categories:
                self.logger.info(f"Found target category: {current_node}")
                self.logger.info(f"Final distance: {current_distance}")
                path = []
                while current_node:
                    path.append(current_node)
                    current_node = previous[current_node]
                path.reverse()
                self.logger.info(f"Found path: {' -> '.join(path)}")
                return (path, current_distance)
            
            # Skip already visited nodes
            if current_node in visited:
                continue
            
            # Mark node as visited
            visited.add(current_node)
            
            # Check neighbors
            for neighbor, base_weight in self.graph[current_node].items():
                # If neighbor already visited, skip
                if neighbor in visited:
                    continue
                
                # Compute weight for this neighbor based on user progress
                adjusted_weight = compute_weight(neighbor) * base_weight
                self.logger.info(f"Adjusted weight from {current_node} to {neighbor}: {adjusted_weight:.2f}")
                
                # Calculate new distance
                distance = current_distance + adjusted_weight
                
                # If we found a shorter path, update distance
                if distance < distances[neighbor]:
                    self.logger.info(f"Updating distance to {neighbor}: {distances[neighbor]:.2f} -> {distance:.2f}")
                    distances[neighbor] = distance
                    previous[neighbor] = current_node
                    heapq.heappush(queue, (distance, neighbor))
        
        # If we get here, no path was found
        self.logger.warning("No path found from start categories to target categories")
        return (None, float('infinity'))
