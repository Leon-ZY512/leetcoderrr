from typing import Dict, List, Optional
from enum import Enum

class Difficulty(Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"

class ProblemNode:
    def __init__(self, name: str, difficulty: Difficulty, category: str, link: str):
        self.name = name
        self.difficulty = difficulty
        self.category = category
        self.link = link
        self.score = 0.0

class DifficultyNode:
    def __init__(self, difficulty: Difficulty):
        self.difficulty = difficulty
        self.children: List[ProblemNode] = []

class ProblemTree:
    def __init__(self):
        self.difficulty_nodes: Dict[Difficulty, DifficultyNode] = {
            Difficulty.EASY: DifficultyNode(Difficulty.EASY),
            Difficulty.MEDIUM: DifficultyNode(Difficulty.MEDIUM),
            Difficulty.HARD: DifficultyNode(Difficulty.HARD)
        }
    
    def add_problem(self, problem: ProblemNode):
        """Add problem to the corresponding difficulty node"""
        self.difficulty_nodes[problem.difficulty].children.append(problem)
    
    def get_problems_by_difficulty(self, difficulty: str) -> List[ProblemNode]:
        """Get all problems of specified difficulty"""
        diff = Difficulty(difficulty)
        return self.difficulty_nodes[diff].children
    
    def get_problems_by_category(self, category: str) -> List[ProblemNode]:
        """Get all problems of specified category"""
        problems = []
        for diff_node in self.difficulty_nodes.values():
            for problem in diff_node.children:
                if problem.category == category:
                    problems.append(problem)
        return problems
    
    def update_problem_score(self, name: str, score: float) -> None:
        """Update problem recommendation score"""
        for diff_node in self.difficulty_nodes.values():
            for problem in diff_node.children:
                if problem.name == name:
                    problem.score = score
                    return
    
    def get_recommended_problems(self, count: int) -> List[ProblemNode]:
        """Get recommended problems based on score from all nodes or by difficulty"""
        all_problems = []
        
        # Collect all nodes
        for diff_name, diff_node in self.difficulty_nodes.items():
            all_problems.extend(diff_node.children)
        
        # Sort by score
        all_problems.sort(key=lambda x: x.score, reverse=True)
        
        # Ensure the returned problem set is appropriate
        # If the first problem is Hard difficulty, check if there's enough experience
        selected_problems = []
        easy_medium_count = 0
        hard_count = 0
        
        for problem in all_problems:
            # Limit the number of Hard problems, ensure the first recommended problem is not Hard
            if problem.difficulty.value == 'Hard':
                hard_count += 1
                if len(selected_problems) == 0 or hard_count > count // 3:
                    continue
            else:
                easy_medium_count += 1
            
            selected_problems.append(problem)
            if len(selected_problems) >= count:
                break
                
        # If not enough recommended problems, add more Easy and Medium problems
        if len(selected_problems) < count:
            remaining_easy_medium = [p for p in all_problems 
                                    if p.difficulty.value != 'Hard' 
                                    and p not in selected_problems]
            remaining_easy_medium.sort(key=lambda x: x.score, reverse=True)
            selected_problems.extend(remaining_easy_medium[:count - len(selected_problems)])
        
        return selected_problems
    
    def get_problem_path(self, name: str) -> List[str]:
        """Get the complete path of the problem (from root to leaf)"""
        path = []
        for diff_node in self.difficulty_nodes.values():
            for problem in diff_node.children:
                if problem.name == name:
                    current = problem
                    while current is not None:
                        path.append(current.name)
                        current = current.parent
                    return list(reversed(path))
        return path
    
    def get_difficulty_progress(self, difficulty: str) -> Dict[str, int]:
        """Get category statistics for the specified difficulty"""
        diff = Difficulty(difficulty)
        category_counts = {}
        for problem in self.difficulty_nodes[diff].children:
            category_counts[problem.category] = category_counts.get(problem.category, 0) + 1
        return category_counts

def build_problem_tree(problems_data: Dict) -> ProblemTree:
    """Build a problem tree from JSON data"""
    tree = ProblemTree()
    
    for category, problems in problems_data.items():
        for problem in problems:
            node = ProblemNode(
                name=problem['name'],
                difficulty=Difficulty(problem['difficulty']),
                category=problem['category'],
                link=problem['link']
            )
            tree.add_problem(node)
    
    return tree 