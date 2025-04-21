import json
from collections import defaultdict
import heapq
import logging

class CategoryGraph:
    def __init__(self):
        # 设置日志记录
        self.logger = self._setup_logger()
        
        # Load problems and extract categories
        with open('problems.json', 'r') as f:
            self.problems = json.load(f)
            self.categories = list(self.problems.keys())
        
        self.logger.info("加载了 %d 个类别的问题", len(self.categories))
        
        # 创建一个从问题名称到类别的映射字典，加速查找
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
        """设置日志记录器"""
        # 创建日志记录器
        logger = logging.getLogger('category_graph')
        logger.setLevel(logging.DEBUG)
        
        # 创建文件处理器
        file_handler = logging.FileHandler('recommendation_log.txt', mode='w')
        file_handler.setLevel(logging.DEBUG)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        logger.addHandler(file_handler)
        
        return logger

    def get_problem_category(self, problem_name):
        """获取问题所属的类别"""
        return self.problem_to_category.get(problem_name)

    def get_next_category_via_dijkstra(self, start_categories, target_categories, user_progress):
        def compute_weight(category):
            data = user_progress['categories'][category]
            attempted = data['total_attempted']
            correct = data['correct_solutions']
            # 如果没有average_time字段，则使用默认值0
            time = data.get('average_time', 0)
            
            # 计算正确率
            accuracy = correct / attempted if attempted > 0 else 0
            normalized_time = time / 1000  # Assuming 1000s is a reasonable max
            
            # 计算该类别下所有已尝试题目的平均得分
            avg_problem_score = 0.0
            problem_scores_count = 0
            
            # 记录调试信息
            self.logger.info(f"计算类别 '{category}' 的权重:")
            self.logger.info(f"  - 正确率: {accuracy:.2f}")
            self.logger.info(f"  - 正规化时间: {normalized_time:.2f}")
            
            # 遍历所有问题，找出属于该类别的问题并计算平均分
            for problem_name, problem_data in user_progress.get('problems', {}).items():
                # 检查问题是否有solutions
                if problem_data.get('solutions', []):
                    # 获取最新的solution评分
                    latest_solution = problem_data['solutions'][-1]
                    score = latest_solution.get('score', 0.0)
                    
                    # 获取问题类别
                    problem_category = self.get_problem_category(problem_name)
                    
                    # 如果题目属于当前类别，加入计算
                    if problem_category == category:
                        self.logger.info(f"  - 问题 '{problem_name}' 得分: {score:.2f}")
                        avg_problem_score += score
                        problem_scores_count += 1
            
            # 计算平均分
            if problem_scores_count > 0:
                avg_problem_score /= problem_scores_count
                self.logger.info(f"  - 该类别平均得分: {avg_problem_score:.2f} (基于 {problem_scores_count} 个问题)")
            else:
                self.logger.info(f"  - 该类别没有已解决的问题")
                avg_problem_score = 0.5  # 对于没有解决问题的类别，使用中等得分
            
            # 新的权重计算公式，考虑正确率、题目平均得分和时间
            weight = (1 - accuracy) * 0.4 + (1 - avg_problem_score) * 0.3 + normalized_time * 0.3
            self.logger.info(f"  - 最终权重: {weight:.2f}")
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
        
        self.logger.info("开始Dijkstra算法寻找最优路径")
        self.logger.info(f"起始类别: {start_categories}")
        self.logger.info(f"目标类别: {target_categories}")
        
        # Dijkstra algorithm
        while queue:
            # Get node with smallest distance
            current_distance, current_node = heapq.heappop(queue)
            
            # If we've reached a target, we're done
            if current_node in target_categories:
                self.logger.info(f"找到目标类别: {current_node}")
                self.logger.info(f"最终距离: {current_distance}")
                path = []
                while current_node:
                    path.append(current_node)
                    current_node = previous[current_node]
                path.reverse()
                self.logger.info(f"找到路径: {' -> '.join(path)}")
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
                self.logger.info(f"从 {current_node} 到 {neighbor} 的调整后权重: {adjusted_weight:.2f}")
                
                # Calculate new distance
                distance = current_distance + adjusted_weight
                
                # If we found a shorter path, update distance
                if distance < distances[neighbor]:
                    self.logger.info(f"更新到 {neighbor} 的距离: {distances[neighbor]:.2f} -> {distance:.2f}")
                    distances[neighbor] = distance
                    previous[neighbor] = current_node
                    heapq.heappush(queue, (distance, neighbor))
        
        # If we get here, no path was found
        self.logger.warning("没有找到从起始类别到目标类别的路径")
        return (None, float('infinity'))
