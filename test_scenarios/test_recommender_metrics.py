import json
import os
import sys
import copy
import random
from pathlib import Path

# 添加父目录到系统路径，以便导入项目模块
sys.path.append(str(Path(__file__).parent.parent))

from recommendation_system import RecommendationSystem
from category_graph import CategoryGraph

class RecommenderMetrics:
    def __init__(self):
        # 备份原始的用户进度文件
        self.backup_user_progress()
        self.recommender = RecommendationSystem()
        self.category_graph = CategoryGraph()
        
        # 加载问题数据
        with open('problems.json', 'r') as f:
            self.problems = json.load(f)
    
    def backup_user_progress(self):
        """备份原始的用户进度文件"""
        if os.path.exists('user_progress.json'):
            with open('user_progress.json', 'r') as f:
                user_progress = json.load(f)
            
            with open('user_progress_backup.json', 'w') as f:
                json.dump(user_progress, f, indent=4)
            print("已备份原始用户进度文件到 user_progress_backup.json")
    
    def restore_user_progress(self):
        """恢复原始的用户进度文件"""
        if os.path.exists('user_progress_backup.json'):
            with open('user_progress_backup.json', 'r') as f:
                user_progress = json.load(f)
            
            with open('user_progress.json', 'w') as f:
                json.dump(user_progress, f, indent=4)
            print("已恢复原始用户进度文件")
    
    def create_test_user(self, scenario_name, user_data):
        """创建测试用户，user_data为字典，包含类别和问题的解决情况"""
        # 读取原始用户进度文件作为模板
        with open('user_progress.json', 'r') as f:
            user_progress = json.load(f)
        
        # 重置所有类别的尝试次数和正确解决次数
        for category in user_progress['user_progress']['categories']:
            user_progress['user_progress']['categories'][category] = {
                'total_attempted': 0,
                'correct_solutions': 0
            }
        
        # 清空所有问题记录
        user_progress['user_progress']['problems'] = {}
        
        # 更新指定类别的尝试次数和正确解决次数
        for category, data in user_data.get('categories', {}).items():
            if category in user_progress['user_progress']['categories']:
                user_progress['user_progress']['categories'][category]['total_attempted'] = data.get('total_attempted', 0)
                user_progress['user_progress']['categories'][category]['correct_solutions'] = data.get('correct_solutions', 0)
        
        # 更新指定问题的解决情况
        for problem_name, data in user_data.get('problems', {}).items():
            user_progress['user_progress']['problems'][problem_name] = {
                'attempts': data.get('attempts', 1),
                'solved': data.get('solved', False),
                'solutions': [
                    {
                        'file_path': f"solutions/{problem_name.replace(' ', '_')}.py",
                        'score': data.get('score', 0.0),
                        'feedback': "Test feedback"
                    }
                ]
            }
        
        # 保存测试用户数据
        scenario_file = f"test_scenarios/{scenario_name}.json"
        with open(scenario_file, 'w') as f:
            json.dump(user_progress, f, indent=4)
        
        print(f"已创建测试用户: {scenario_name}")
        return scenario_file
    
    def apply_test_user(self, scenario_file):
        """应用测试用户数据"""
        if os.path.exists(scenario_file):
            with open(scenario_file, 'r') as f:
                user_progress = json.load(f)
            
            with open('user_progress.json', 'w') as f:
                json.dump(user_progress, f, indent=4)
            
            # 重新初始化推荐系统以加载新的用户数据
            self.recommender = RecommendationSystem()
            print(f"已应用测试用户: {scenario_file}")
        else:
            print(f"测试用户文件不存在: {scenario_file}")
    
    def simulate_user_progress(self, num_steps, success_rate=0.7):
        """模拟用户进度，获取推荐问题并随机解决"""
        print(f"\n=== 模拟用户进度 ({num_steps}步) ===")
        recommendations_history = []
        
        for step in range(1, num_steps + 1):
            print(f"\n步骤 {step}:")
            
            # 获取推荐
            recommendations = self.recommender.get_recommendations(3)
            recommendations_history.append(recommendations)
            
            # 随机选择一个推荐问题
            selected_problem = random.choice(recommendations)
            print(f"选择问题: {selected_problem['name']} ({selected_problem['difficulty']})")
            
            # 随机决定是否解决成功
            is_solved = random.random() < success_rate
            
            # 更新进度
            self.update_user_progress(selected_problem, is_solved)
            
            if is_solved:
                print(f"✓ 成功解决!")
            else:
                print(f"✗ 未能解决")
            
            # 分析当前用户状态
            self.analyze_user_state()
        
        return recommendations_history
    
    def update_user_progress(self, problem, is_solved):
        """更新用户进度"""
        # 获取问题类别
        problem_category = problem['category']
        
        # 读取当前用户进度
        with open('user_progress.json', 'r') as f:
            user_progress = json.load(f)
        
        # 更新类别统计
        if problem_category in user_progress['user_progress']['categories']:
            user_progress['user_progress']['categories'][problem_category]['total_attempted'] += 1
            if is_solved:
                user_progress['user_progress']['categories'][problem_category]['correct_solutions'] += 1
        
        # 更新问题记录
        if problem['name'] not in user_progress['user_progress']['problems']:
            user_progress['user_progress']['problems'][problem['name']] = {
                'attempts': 0,
                'solved': False,
                'solutions': []
            }
        
        problem_stats = user_progress['user_progress']['problems'][problem['name']]
        problem_stats['attempts'] += 1
        if is_solved:
            problem_stats['solved'] = True
        
        # 添加解决方案记录
        score = 0.8 if is_solved else 0.3
        problem_stats['solutions'].append({
            'file_path': f"solutions/{problem['name'].replace(' ', '_')}.py",
            'score': score,
            'feedback': "Simulated feedback"
        })
        
        # 保存更新后的进度
        with open('user_progress.json', 'w') as f:
            json.dump(user_progress, f, indent=4)
        
        # 重新初始化推荐系统以加载新的进度数据
        self.recommender = RecommendationSystem()
    
    def analyze_user_state(self):
        """分析当前用户状态"""
        # 获取用户强弱类别
        strong_categories = self.recommender.get_user_strong_categories()
        weak_categories = self.recommender.get_user_weak_categories()
        
        print("\n当前用户状态:")
        print(f"强类别: {', '.join(strong_categories) if strong_categories else '无'}")
        print(f"弱类别: {', '.join(weak_categories) if weak_categories else '无'}")
        
        # 计算总尝试次数和成功率
        total_attempted = 0
        total_correct = 0
        for stats in self.recommender.progress['user_progress']['categories'].values():
            total_attempted += stats['total_attempted']
            total_correct += stats['correct_solutions']
        
        if total_attempted > 0:
            overall_success_rate = total_correct / total_attempted * 100
            print(f"总体成功率: {overall_success_rate:.1f}% ({total_correct}/{total_attempted})")
    
    def calculate_coverage_metrics(self, recommendations_history):
        """计算推荐覆盖率指标"""
        print("\n=== 推荐覆盖率指标 ===")
        
        # 统计推荐的独特问题数
        unique_problems = set()
        for recommendations in recommendations_history:
            for rec in recommendations:
                unique_problems.add(rec['name'])
        
        # 统计推荐的独特类别数
        unique_categories = set()
        for recommendations in recommendations_history:
            for rec in recommendations:
                unique_categories.add(rec['category'])
        
        # 计算推荐的独特难度分布
        difficulty_counts = {'Easy': 0, 'Medium': 0, 'Hard': 0}
        for recommendations in recommendations_history:
            for rec in recommendations:
                difficulty_counts[rec['difficulty']] += 1
        
        # 计算覆盖率
        total_problems = sum(len(problems) for problems in self.problems.values())
        total_categories = len(self.problems)
        
        problem_coverage = len(unique_problems) / total_problems * 100
        category_coverage = len(unique_categories) / total_categories * 100
        
        print(f"问题覆盖率: {problem_coverage:.1f}% ({len(unique_problems)}/{total_problems})")
        print(f"类别覆盖率: {category_coverage:.1f}% ({len(unique_categories)}/{total_categories})")
        
        print("\n难度分布:")
        total_recs = sum(difficulty_counts.values())
        for diff, count in difficulty_counts.items():
            percentage = count / total_recs * 100
            print(f"  {diff}: {count} 次推荐 ({percentage:.1f}%)")
    
    def calculate_novelty_diversity(self, recommendations_history):
        """计算推荐新颖性和多样性指标"""
        print("\n=== 推荐新颖性和多样性指标 ===")
        
        # 新颖性: 计算平均每轮推荐中首次出现的问题比例
        first_appearances = set()
        novelty_scores = []
        
        for i, recommendations in enumerate(recommendations_history):
            new_in_this_round = 0
            for rec in recommendations:
                if rec['name'] not in first_appearances:
                    first_appearances.add(rec['name'])
                    new_in_this_round += 1
            
            novelty_scores.append(new_in_this_round / len(recommendations))
        
        avg_novelty = sum(novelty_scores) / len(novelty_scores) * 100 if novelty_scores else 0
        
        # 多样性: 计算平均每轮推荐中不同类别和难度的比例
        diversity_scores = []
        
        for recommendations in recommendations_history:
            categories = set(rec['category'] for rec in recommendations)
            difficulties = set(rec['difficulty'] for rec in recommendations)
            
            # 类别多样性和难度多样性的平均值
            cat_diversity = len(categories) / len(recommendations)
            diff_diversity = len(difficulties) / len(recommendations)
            
            diversity_scores.append((cat_diversity + diff_diversity) / 2)
        
        avg_diversity = sum(diversity_scores) / len(diversity_scores) * 100 if diversity_scores else 0
        
        print(f"平均新颖性: {avg_novelty:.1f}%")
        print(f"平均多样性: {avg_diversity:.1f}%")
    
    def evaluate_recommendation_quality(self):
        """测试推荐系统在各种场景下的质量"""
        try:
            # 创建新手用户
            beginner_data = {
                'categories': {},
                'problems': {}
            }
            beginner_file = self.create_test_user("metrics_beginner", beginner_data)
            self.apply_test_user(beginner_file)
            
            # 模拟用户进度10步
            print("\n=== 新手用户推荐测试 ===")
            recs_history = self.simulate_user_progress(10)
            
            # 计算指标
            self.calculate_coverage_metrics(recs_history)
            self.calculate_novelty_diversity(recs_history)
            
            # 创建有经验用户
            experienced_data = {
                'categories': {
                    'Arrays and Hashing': {'total_attempted': 10, 'correct_solutions': 8},
                    'Two Pointers': {'total_attempted': 8, 'correct_solutions': 6},
                    'Stack': {'total_attempted': 5, 'correct_solutions': 3},
                    'Binary Search': {'total_attempted': 4, 'correct_solutions': 2}
                },
                'problems': {
                    'Two Sum': {'attempts': 1, 'solved': True, 'score': 0.9},
                    'Valid Anagram': {'attempts': 1, 'solved': True, 'score': 0.85},
                    'Contains Duplicate': {'attempts': 1, 'solved': True, 'score': 0.95},
                    'Valid Parentheses': {'attempts': 2, 'solved': False, 'score': 0.4},
                    'Binary Search': {'attempts': 3, 'solved': True, 'score': 0.7}
                }
            }
            experienced_file = self.create_test_user("metrics_experienced", experienced_data)
            self.apply_test_user(experienced_file)
            
            # 模拟用户进度10步
            print("\n=== 有经验用户推荐测试 ===")
            recs_history = self.simulate_user_progress(10)
            
            # 计算指标
            self.calculate_coverage_metrics(recs_history)
            self.calculate_novelty_diversity(recs_history)
            
        finally:
            # 恢复原始用户进度
            self.restore_user_progress()

if __name__ == "__main__":
    metrics = RecommenderMetrics()
    metrics.evaluate_recommendation_quality() 