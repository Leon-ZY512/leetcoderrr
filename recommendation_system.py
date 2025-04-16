import json
import random
import os
from typing import List, Dict, Tuple
from problem_tree import ProblemTree, build_problem_tree
import openai
import re
from category_graph import CategoryGraph
from collections import defaultdict
from substring_search import kmp_search, search_problems

# 从.env文件读取环境变量
def load_env():
    try:
        with open('.env', 'r') as f:
            content = f.read()
            # 直接查找OPENAI_API_KEY
            api_key_match = re.search(r'OPENAI_API_KEY="(.*?)"', content, re.MULTILINE)
            if api_key_match:
                key = api_key_match.group(1).strip()
                # 移除末尾可能存在的百分号
                if key.endswith('%'):
                    key = key[:-1]
                os.environ["OPENAI_API_KEY"] = key
                print("成功读取API密钥")
            else:
                print("警告: 在.env文件中未找到OPENAI_API_KEY")
    except FileNotFoundError:
        print("警告: .env文件未找到")

# 加载环境变量
load_env()

class RecommendationSystem:
    def __init__(self):
        with open('problems.json', 'r') as f:
            self.problems = json.load(f)
        with open('user_progress.json', 'r') as f:
            self.progress = json.load(f)
        self.problem_tree = build_problem_tree(self.problems)
        self.category_graph = CategoryGraph()
        
        # 创建solutions目录
        if not os.path.exists('solutions'):
            os.makedirs('solutions')
    
    def generate_solution_file(self, problem: Dict) -> str:
        """Generate solution file for the selected problem"""
        # Create filename (remove spaces and special characters)
        file_name = ''.join(c for c in problem['name'] if c.isalnum() or c == ' ').replace(' ', '_')
        file_path = f"solutions/{file_name}.py"
        
        # Generate file content
        content = f'''"""
{problem['name']}
Difficulty: {problem['difficulty']}
Category: {problem['category']}
Link: {problem['link']}
"""

def solution(nums):
    """
    Implement your solution here
    """
    pass

if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Add your test cases here
    ]
    
    for test_case in test_cases:
        result = solution(test_case)
        print(f"Input: {{{{test_case}}}}")
        print(f"Output: {{{{result}}}}")
'''
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write(content)
        
        return file_path
    
    def evaluate_solution(self, file_path: str) -> Dict:
        """使用GPT评估解决方案"""
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            # 从环境变量获取API密钥
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return {
                    "score": 0.0,
                    "feedback": "API key not found. Please set the OPENAI_API_KEY environment variable."
                }
            
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """You are a professional algorithm code reviewer. Please evaluate the quality of the following solution.

Scoring criteria:
1. Correctness: Does the solution correctly solve the problem (4 points)
2. Edge Case Handling: Does the solution properly handle all edge cases (2 points)
3. Time Complexity: Does it achieve optimal or near-optimal time complexity for this problem (2 points)
4. Space Complexity: Does it achieve optimal or near-optimal space complexity for this problem (2 points)

Please ignore the following factors:
- Completeness and diversity of test cases
- Code style and naming conventions (unless they severely impact readability)

At the end of your evaluation, provide a total score (out of 10), formatted as: 'Final Score: X/10'
                    """},
                    {"role": "user", "content": f"Please evaluate this LeetCode solution code:\n\n{code}"}
                ]
            )
            
            feedback = response.choices[0].message.content
            
            # 解析GPT的反馈，提取分数
            score = 0.0
            if "Final Score" in feedback or "final score" in feedback:
                try:
                    score_matches = re.findall(r'(?:Final Score|final score).*?(\d+\.?\d*)\s*\/\s*(\d+\.?\d*)', feedback, re.IGNORECASE)
                    if score_matches:
                        numerator, denominator = score_matches[0]
                        score = float(numerator) / float(denominator)  # 归一化到0-1范围
                except:
                    pass
            # 尝试从其他评价中提取分数
            elif "score" in feedback.lower():
                try:
                    # 查找包含数字/数字格式的评分
                    score_matches = re.findall(r'score.*?(\d+\.?\d*)\s*\/\s*(\d+\.?\d*)', feedback.lower(), re.IGNORECASE)
                    if score_matches:
                        numerator, denominator = score_matches[0]
                        score = float(numerator) / float(denominator)  # 归一化到0-1范围
                except:
                    pass
            
            return {
                "score": score,
                "feedback": feedback
            }
            
        except Exception as e:
            return {
                "score": 0.0,
                "feedback": f"评估过程中出现错误：{str(e)}"
            }
    
    def save_progress(self):
        """保存用户进度到文件"""
        with open('user_progress.json', 'w') as f:
            json.dump(self.progress, f, indent=4)
    
    def submit_solution(self, problem_name: str, file_path: str):
        """Submit problem solution"""
        # Find the problem category
        problem_category = None
        for category, problems in self.problems.items():
            for problem in problems:
                if problem['name'] == problem_name:
                    problem_category = problem['category']
                    break
            if problem_category:
                break
        
        if not problem_category:
            print("Problem not found")
            return
        
        # Evaluate solution
        evaluation = self.evaluate_solution(file_path)
        is_correct = evaluation['score'] >= 0.7  # Score of 70% or higher is considered correct
        
        # Update category statistics
        category_stats = self.progress['user_progress']['categories'][problem_category]
        category_stats['total_attempted'] += 1
        if is_correct:
            category_stats['correct_solutions'] += 1
        
        # Update problem record
        if problem_name not in self.progress['user_progress']['problems']:
            self.progress['user_progress']['problems'][problem_name] = {
                'attempts': 0,
                'solved': False,
                'solutions': []
            }
        
        problem_stats = self.progress['user_progress']['problems'][problem_name]
        problem_stats['attempts'] += 1
        if is_correct:
            problem_stats['solved'] = True
        
        # Record solution
        problem_stats['solutions'].append({
            'file_path': file_path,
            'score': evaluation['score'],
            'feedback': evaluation['feedback']
        })
        
        # Save progress
        self.save_progress()
        
        # Print feedback
        print("\n=== Code Evaluation Results ===")
        print(f"Score: {evaluation['score']:.1f}/1.0")
        print("\nDetailed Feedback:")
        print(evaluation['feedback'])
        
        if is_correct:
            print(f"\nCongratulations! You have successfully solved {problem_name}!")
            success_rate = category_stats['correct_solutions'] / category_stats['total_attempted'] * 100
            print(f"Your success rate in {problem_category} category: {success_rate:.1f}%")
        else:
            print(f"\nKeep going! This is your attempt #{problem_stats['attempts']} for {problem_name}")
            if category_stats['total_attempted'] > 1:
                success_rate = category_stats['correct_solutions'] / category_stats['total_attempted'] * 100
                print(f"Your success rate in {problem_category} category: {success_rate:.1f}%")
    
    def is_initial_user(self) -> bool:
        """Check if this is the first time user is using the system"""
        total_attempts = sum(
            stats['total_attempted'] 
            for stats in self.progress['user_progress']['categories'].values()
        )
        return total_attempts == 0
    
    def calculate_category_score(self, category: str) -> float:
        """计算分类的推荐优先级分数"""
        stats = self.progress['user_progress']['categories'].get(category, {
            'total_attempted': 0,
            'correct_solutions': 0
        })
        
        if stats['total_attempted'] == 0:
            return 1.0  # 未尝试过的分类优先级最高
        
        success_rate = stats['correct_solutions'] / stats['total_attempted']
        attempt_count = stats['total_attempted']
        
        # 计算分数：考虑正确率和尝试次数
        # 正确率越低，分数越高（需要加强）
        # 尝试次数越多，分数越低（避免重复）
        score = (1 - success_rate) * (1 / (1 + attempt_count * 0.1))
        return score
    
    def calculate_problem_score(self, problem: Dict, category: str) -> float:
        """Calculate problem score based on user progress"""
        # Calculate base score based on difficulty
        if problem['difficulty'] == 'Easy':
            base_score = 3.0
        elif problem['difficulty'] == 'Medium':
            base_score = 2.0
        else:  # Hard
            base_score = 0.5  # Significantly lower base score for Hard problems
        
        # Get total problems attempted across all categories
        total_problems_attempted = sum(
            stats['total_attempted'] 
            for stats in self.progress['user_progress']['categories'].values()
        )
        
        # If user hasn't attempted many problems yet, prioritize Easy and Medium problems in weak categories
        if total_problems_attempted < 10:  # Beginner user
            # Hard problems should be almost never recommended for beginners
            if problem['difficulty'] == 'Hard':
                base_score *= 0.1  # Very low score for Hard problems
            
            # Prioritize problems from categories where user has low performance
            if category in self.progress['user_progress']['categories']:
                category_stats = self.progress['user_progress']['categories'][category]
                if category_stats['total_attempted'] > 0:
                    success_rate = category_stats['correct_solutions'] / category_stats['total_attempted']
                    # Lower success rate = higher priority
                    category_factor = 1.0 + (1.0 - success_rate)
                    base_score *= category_factor
                else:
                    # Boost score for categories user hasn't tried yet
                    base_score *= 1.5
        else:  # More experienced user
            if category in self.progress['user_progress']['categories']:
                category_stats = self.progress['user_progress']['categories'][category]
                if category_stats['total_attempted'] > 0:
                    success_rate = category_stats['correct_solutions'] / category_stats['total_attempted']
                    
                    # Adjust based on success rate and difficulty
                    if success_rate < 0.5:  # Less than 50% success rate
                        if problem['difficulty'] == 'Easy':
                            base_score *= 1.5  # Prioritize Easy problems for struggling users
                        elif problem['difficulty'] == 'Medium':
                            base_score *= 1.0  # Normal priority for Medium
                        else:  # Hard
                            base_score *= 0.3  # Low priority for Hard
                    elif success_rate < 0.7:  # Between 50% and 70% success rate
                        if problem['difficulty'] == 'Easy':
                            base_score *= 1.0  # Normal for Easy
                        elif problem['difficulty'] == 'Medium':
                            base_score *= 1.2  # Slightly higher for Medium
                        else:  # Hard
                            base_score *= 0.5  # Still low for Hard
                    else:  # Success rate >= 70%
                        if problem['difficulty'] == 'Easy':
                            base_score *= 0.8  # Lower priority for Easy
                        elif problem['difficulty'] == 'Medium':
                            base_score *= 1.3  # Higher for Medium
                        else:  # Hard
                            base_score *= 1.0  # Normal for Hard, but only for successful users
                else:
                    # Prioritize unexplored categories
                    base_score *= 1.5
                    
                    # Still limit Hard problems for new categories
                    if problem['difficulty'] == 'Hard':
                        base_score *= 0.6
                        
        # Check if user has already attempted this problem
        if problem['name'] in self.progress['user_progress']['problems']:
            problem_stats = self.progress['user_progress']['problems'][problem['name']]
            
            # If already solved, reduce score significantly
            if problem_stats['solved']:
                base_score *= 0.1
            else:
                # If attempted but not solved, increase score slightly to encourage completion
                base_score *= 1.2
                
        return base_score
    
    def get_initial_problems(self, count: int = 3) -> List[Dict]:
        """Recommend initial problems for first-time users from diverse categories"""
        initial_problems = []
        
        # Get all unique categories
        all_categories = set()
        for category_problems in self.problems.values():
            for problem in category_problems:
                all_categories.add(problem['category'])
        
        # Create a dictionary of easy problems by category
        easy_problems_by_category = {}
        medium_problems_by_category = {}
        
        for category_name, problems_list in self.problems.items():
            for problem in problems_list:
                category = problem['category']
                if problem['difficulty'] == 'Easy':
                    if category not in easy_problems_by_category:
                        easy_problems_by_category[category] = []
                    easy_problems_by_category[category].append(problem)
                elif problem['difficulty'] == 'Medium':
                    if category not in medium_problems_by_category:
                        medium_problems_by_category[category] = []
                    medium_problems_by_category[category].append(problem)
        
        # Get a diverse selection of categories
        category_list = list(all_categories)
        random.shuffle(category_list)
        selected_categories = category_list[:min(count, len(category_list))]
        
        # Try to get one easy problem from each selected category
        for category in selected_categories:
            if category in easy_problems_by_category and easy_problems_by_category[category]:
                # Select a random easy problem from this category
                problem = random.choice(easy_problems_by_category[category])
                initial_problems.append({
                    'name': problem['name'],
                    'difficulty': problem['difficulty'],
                    'category': problem['category'],
                    'link': problem['link'],
                    'reason': f"This is an Easy problem in {problem['category']}, recommended to start your algorithm learning journey."
                })
            elif category in medium_problems_by_category and medium_problems_by_category[category]:
                # If no easy problem in this category, try a medium one
                problem = random.choice(medium_problems_by_category[category])
                initial_problems.append({
                    'name': problem['name'],
                    'difficulty': problem['difficulty'],
                    'category': problem['category'],
                    'link': problem['link'],
                    'reason': f"This is a Medium problem in {problem['category']}, recommended to develop your algorithm skills."
                })
        
        # If we don't have enough problems yet, add more from other categories
        if len(initial_problems) < count:
            remaining_count = count - len(initial_problems)
            used_categories = {prob['category'] for prob in initial_problems}
            remaining_categories = [cat for cat in category_list if cat not in used_categories]
            
            for category in remaining_categories[:remaining_count]:
                if category in easy_problems_by_category and easy_problems_by_category[category]:
                    problem = random.choice(easy_problems_by_category[category])
                    initial_problems.append({
                        'name': problem['name'],
                        'difficulty': problem['difficulty'],
                        'category': problem['category'],
                        'link': problem['link'],
                        'reason': f"This is an Easy problem in {problem['category']}, recommended to broaden your algorithm knowledge."
                    })
                elif category in medium_problems_by_category and medium_problems_by_category[category]:
                    problem = random.choice(medium_problems_by_category[category])
                    initial_problems.append({
                        'name': problem['name'],
                        'difficulty': problem['difficulty'],
                        'category': problem['category'],
                        'link': problem['link'],
                        'reason': f"This is a Medium problem in {problem['category']}, recommended for advanced algorithm practice."
                    })
        
        return initial_problems
    
    def get_user_weak_categories(self) -> List[str]:
        """找出用户表现较弱的分类"""
        weak_categories = []
        for category, stats in self.progress['user_progress']['categories'].items():
            if stats['total_attempted'] > 0:
                success_rate = stats['correct_solutions'] / stats['total_attempted']
                if success_rate < 0.6:  # 正确率低于60%的分类
                    weak_categories.append(category)
        return weak_categories
    
    def get_user_strong_categories(self) -> List[str]:
        """找出用户表现较强的分类"""
        strong_categories = []
        for category, stats in self.progress['user_progress']['categories'].items():
            if stats['total_attempted'] > 0:
                success_rate = stats['correct_solutions'] / stats['total_attempted']
                if success_rate >= 0.8:  # 正确率高于80%的分类
                    strong_categories.append(category)
        return strong_categories
    
    def get_recommendations(self, count: int = 3) -> List[Dict]:
        """Recommend problems based on user performance in algorithm categories"""
        # If first-time user with no attempts, return initial easy problems
        if self.is_initial_user():
            return self.get_initial_problems(count)
        
        # Get a list of all unique categories
        all_categories = set()
        for category_problems in self.problems.values():
            for problem in category_problems:
                all_categories.add(problem['category'])
        
        # Get categories the user has already attempted
        attempted_categories = set()
        for category, stats in self.progress['user_progress']['categories'].items():
            if stats['total_attempted'] > 0:
                attempted_categories.add(category)
        
        # If user hasn't tried all categories yet, prioritize unexplored categories
        if len(attempted_categories) < len(all_categories):
            return self.get_unexplored_category_problems(all_categories, attempted_categories, count)
        
        # If user has tried all categories, use regular recommendation logic
        return self.get_performance_based_recommendations(count)
    
    def get_unexplored_category_problems(self, all_categories: set, attempted_categories: set, count: int) -> List[Dict]:
        """Recommend problems from unexplored categories"""
        recommendations = []
        
        # Get unexplored categories
        unexplored_categories = all_categories - attempted_categories
        
        # First, try to get problems from unexplored categories (prioritizing Easy difficulty)
        easy_problems_by_category = {}
        medium_problems_by_category = {}
        
        # Collect all Easy and Medium problems by category
        for category_name, category_problems in self.problems.items():
            for problem in category_problems:
                if problem['category'] in unexplored_categories:
                    if problem['name'] not in self.progress['user_progress']['problems']:
                        if problem['difficulty'] == 'Easy':
                            if problem['category'] not in easy_problems_by_category:
                                easy_problems_by_category[problem['category']] = []
                            easy_problems_by_category[problem['category']].append(problem)
                        elif problem['difficulty'] == 'Medium':
                            if problem['category'] not in medium_problems_by_category:
                                medium_problems_by_category[problem['category']] = []
                            medium_problems_by_category[problem['category']].append(problem)
        
        # First try to add one Easy problem from each unexplored category
        for category in unexplored_categories:
            if len(recommendations) >= count:
                break
                
            if category in easy_problems_by_category and easy_problems_by_category[category]:
                problem = random.choice(easy_problems_by_category[category])
                recommendations.append({
                    'name': problem['name'],
                    'difficulty': problem['difficulty'],
                    'category': problem['category'],
                    'link': problem['link'],
                    'reason': f"This is an Easy problem in {problem['category']}, which you haven't explored yet."
                })
        
        # If we still need more recommendations, add Medium problems from unexplored categories
        if len(recommendations) < count:
            for category in unexplored_categories:
                if len(recommendations) >= count:
                    break
                    
                if category in medium_problems_by_category and medium_problems_by_category[category]:
                    # Skip categories that already have an Easy problem recommended
                    if any(rec['category'] == category for rec in recommendations):
                        continue
                        
                    problem = random.choice(medium_problems_by_category[category])
                    recommendations.append({
                        'name': problem['name'],
                        'difficulty': problem['difficulty'],
                        'category': problem['category'],
                        'link': problem['link'],
                        'reason': f"This is a Medium problem in {problem['category']}, which you haven't explored yet."
                    })
        
        # If we still need more, add problems from already explored categories with low attempt count
        if len(recommendations) < count:
            # Sort categories by attempt count
            sorted_categories = sorted(
                attempted_categories,
                key=lambda cat: self.progress['user_progress']['categories'][cat]['total_attempted']
            )
            
            # Add problems from least attempted categories
            for category in sorted_categories:
                if len(recommendations) >= count:
                    break
                
                # Skip categories already in recommendations
                if any(rec['category'] == category for rec in recommendations):
                    continue
                
                # Try to find an Easy or Medium problem in this category
                category_problems = []
                for cat_name, cat_problems in self.problems.items():
                    for problem in cat_problems:
                        if problem['category'] == category and problem['difficulty'] != 'Hard':
                            if problem['name'] not in self.progress['user_progress']['problems']:
                                category_problems.append(problem)
                
                if category_problems:
                    problem = random.choice(category_problems)
                    attempt_count = self.progress['user_progress']['categories'][category]['total_attempted']
                    recommendations.append({
                        'name': problem['name'],
                        'difficulty': problem['difficulty'],
                        'category': problem['category'],
                        'link': problem['link'],
                        'reason': f"This problem will help you strengthen your knowledge in {category}, which you've only attempted {attempt_count} times."
                    })
        
        return recommendations
    
    def get_performance_based_recommendations(self, count: int) -> List[Dict]:
        """Recommend problems based on performance after all categories have been explored"""
        recommendations = []
        all_scored_problems = []
        max_per_category = 2
        category_counts = defaultdict(int)
        
        # use graph to identify next priority category
        Strong_categories = self.get_user_strong_categories()
        Weak_categories = self.get_user_weak_categories()
        next_category = self.category_graph.get_next_category_via_dijkstra(
            start_categories=Strong_categories or list(self.progress['user_progress']['categories'].keys()),
            target_categories=Weak_categories,
            user_progress=self.progress['user_progress']
        )

        # Calculate total attempted problems
        total_attempted = sum(
            stats['total_attempted'] 
            for stats in self.progress['user_progress']['categories'].values()
        )
        
        # Calculate scores for all problems
        for diff_node in self.problem_tree.difficulty_nodes.values():
            for problem in diff_node.children:
                # Skip Hard problems for beginners (users with < 14 problems attempted)
                if problem.difficulty.value == 'Hard' and total_attempted < 14:
                    continue
                    
                score = self.calculate_problem_score({
                    'name': problem.name,
                    'difficulty': problem.difficulty.value,
                    'category': problem.category,
                    'link': problem.link
                }, problem.category)
                
                if problem.category == next_category:
                    score *= 2.0

                problem.score = score
                all_scored_problems.append(problem)
        
        # Sort all problems by score
        all_scored_problems.sort(key=lambda x: x.score, reverse=True)
        
        # Ensure category diversity in recommendations
        categories_seen = set()
        diverse_recommendations = []
        
        # First pass: try to get diverse categories
        for problem in all_scored_problems:
            if len(diverse_recommendations) >= count:
                break
                
            # Skip problems already solved
            if problem.name in self.progress['user_progress']['problems'] and \
               self.progress['user_progress']['problems'][problem.name]['solved']:
                continue
            
            # limit each category to max_per_category problems
            if category_counts[problem.category] >= max_per_category:
                continue
            
            # If we haven't seen this category yet or it's a high-scoring problem
            if problem.category not in categories_seen or problem.score > 2.0:
                diverse_recommendations.append(problem)
                categories_seen.add(problem.category)
                category_counts[problem.category] += 1

        # Second pass: if we need more recommendations, add highest scoring remaining problems
        if len(diverse_recommendations) < count:
            # Filter out problems already selected and ones already solved
            remaining_problems = [
                p for p in all_scored_problems 
                if p not in diverse_recommendations and 
                (p.name not in self.progress['user_progress']['problems'] or
                 not self.progress['user_progress']['problems'][p.name]['solved'])
            ]
            
            # Add remaining problems up to count
            diverse_recommendations.extend(remaining_problems[:count - len(diverse_recommendations)])
        
        # Format recommendations
        for problem in diverse_recommendations:
            # Create reason based on problem attributes
            if problem.name in self.progress['user_progress']['problems'] and not self.progress['user_progress']['problems'][problem.name]['solved']:
                reason = f"You've attempted this {problem.difficulty.value} problem before but haven't solved it yet."
            elif problem.category in categories_seen and any(stats['total_attempted'] > 0 for cat, stats in self.progress['user_progress']['categories'].items() if cat == problem.category):
                category_stats = self.progress['user_progress']['categories'][problem.category]
                if category_stats['total_attempted'] > 0:
                    success_rate = category_stats['correct_solutions'] / category_stats['total_attempted'] * 100
                    if success_rate < 50:
                        reason = f"This {problem.difficulty.value} problem will help improve your skills in {problem.category} (current success rate: {success_rate:.1f}%)."
                    else:
                        reason = f"Based on your {success_rate:.1f}% success rate in {problem.category}, this {problem.difficulty.value} problem should be appropriate."
                else:
                    reason = f"This {problem.difficulty.value} problem will introduce you to {problem.category}."
            else:
                reason = f"This {problem.difficulty.value} problem in {problem.category} is recommended based on your overall progress."
            
            recommendations.append({
                'name': problem.name,
                'difficulty': problem.difficulty.value,
                'category': problem.category,
                'link': problem.link,
                'reason': reason
            })
        
        return recommendations

def main():
    recommender = RecommendationSystem()
    current_problem = None
    current_file = None
    
    while True:
        print("\n=== LeetCode Recommendation System ===")
        if recommender.is_initial_user():
            print("Welcome to the LeetCode Recommendation System!")
            print("We'll recommend some beginner problems to help you start your algorithm learning journey.")
        
        print("\n1. Get Recommended Problems")
        print("2. View My Weak Areas")
        print("3. View My Strong Areas")
        print("4. View Difficulty Distribution")
        print("5. Submit Current Problem")
        print("6. Search Problems")
        print("7. Exit")
        
        choice = input("\nSelect an option (1-7): ")
        
        if choice == '1':
            recommendations = recommender.get_recommendations()
            print("\n=== Recommended Problems ===")
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['name']}")
                print(f"   Difficulty: {rec['difficulty']}")
                print(f"   Category: {rec['category']}")
                print(f"   Link: {rec['link']}")
                print(f"   Reason: {rec['reason']}")
            
            # Choose a problem
            while True:
                try:
                    choice = int(input("\nSelect a problem (1-3), or enter 0 to return: "))
                    if choice == 0:
                        break
                    if 1 <= choice <= len(recommendations):
                        current_problem = recommendations[choice-1]
                        current_file = recommender.generate_solution_file(current_problem)
                        print(f"\nSelected problem: {current_problem['name']}")
                        print(f"Solution file generated: {current_file}")
                        print("Complete the code and use option 5 to submit your solution")
                        break
                    else:
                        print("Invalid choice, please try again")
                except ValueError:
                    print("Please enter a valid number")
        
        elif choice == '2':
            weak_categories = recommender.get_user_weak_categories()
            print("\n=== Areas to Improve ===")
            if not weak_categories:
                print("No data yet or you're doing well in all categories!")
            for category in weak_categories:
                stats = recommender.progress['user_progress']['categories'][category]
                success_rate = (stats['correct_solutions'] / stats['total_attempted']) * 100
                print(f"\n{category}:")
                print(f"  Attempts: {stats['total_attempted']}")
                print(f"  Success Rate: {success_rate:.1f}%")
        
        elif choice == '3':
            strong_categories = recommender.get_user_strong_categories()
            print("\n=== Strong Areas ===")
            if not strong_categories:
                print("No data yet or keep practicing!")
            for category in strong_categories:
                stats = recommender.progress['user_progress']['categories'][category]
                success_rate = (stats['correct_solutions'] / stats['total_attempted']) * 100
                print(f"\n{category}:")
                print(f"  Attempts: {stats['total_attempted']}")
                print(f"  Success Rate: {success_rate:.1f}%")
        
        elif choice == '4':
            print("\n=== Problem Difficulty Distribution ===")
            for difficulty in ['Easy', 'Medium', 'Hard']:
                problems = recommender.problem_tree.get_problems_by_difficulty(difficulty)
                categories = recommender.problem_tree.get_difficulty_progress(difficulty)
                print(f"\n{difficulty} Difficulty:")
                for category, count in categories.items():
                    print(f"  {category}: {count} problems")
        
        elif choice == '5':
            if current_problem is None or current_file is None:
                print("\nPlease select a problem first!")
                continue
            
            print(f"\nCurrent problem: {current_problem['name']}")
            print(f"Solution file: {current_file}")
            
            if not os.path.exists(current_file):
                print("Solution file not found!")
                continue
            
            while True:
                confirm = input("Are you sure you want to submit this solution? (y/n): ").lower()
                if confirm in ['y', 'n']:
                    if confirm == 'y':
                        recommender.submit_solution(current_problem['name'], current_file)
                    current_problem = None
                    current_file = None
                    break
                print("Please enter y or n")
        
        elif choice == '6':
            keyword = input("Enter a keyword to search for problems: ")
            if not keyword:
                continue

            results = search_problems(keyword)

            if not results:
                print(f"No problems found containing '{keyword}'")
                continue

            print(f"\nFound {len(results)} problems:")
            for i, problem in enumerate(results, 1):
                print(f"\n{i}. {problem['name']}")
                print(f"   difficulty: {problem['difficulty']}")
                print(f"   category: {problem['category']}")
                print(f"   link: {problem['link']}")

            while True:
                try:
                    choice = int(input("\nSelect a problem (1-{}), or enter 0 to return: ".format(len(results))))
                    if choice == 0:
                        break
                    if 1 <= choice <= len(results):
                        current_problem = results[choice-1]
                        current_file = recommender.generate_solution_file(current_problem)
                        print(f"\nSelected problem: {current_problem['name']}")
                        print(f"Solution file generated: {current_file}")
                        print("Complete the code and use option 5 to submit your solution")
                        break
                    else:
                        print("Invalid choice, please try again")
                except ValueError:
                    print("Please enter a valid number")

        elif choice == '7':
            print("\nThank you for using the system! Goodbye!")
            break
        
        else:
            print("Invalid choice, please try again")

if __name__ == "__main__":
    main() 