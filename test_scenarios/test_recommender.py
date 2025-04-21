import json
import os
import sys
import copy
from pathlib import Path

# 添加父目录到系统路径，以便导入项目模块
sys.path.append(str(Path(__file__).parent.parent))

from recommendation_system import RecommendationSystem
from category_graph import CategoryGraph

class RecommenderTester:
    def __init__(self):
        # 备份原始的用户进度文件
        self.backup_user_progress()
        self.recommender = RecommendationSystem()
        self.category_graph = CategoryGraph()
    
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
    
    def create_test_scenario(self, scenario_name, categories_data):
        """创建测试场景的用户进度数据"""
        # 读取原始用户进度文件作为模板
        with open('user_progress.json', 'r') as f:
            user_progress = json.load(f)
        
        # 修改用户进度数据
        for category, data in categories_data.items():
            if category in user_progress['user_progress']['categories']:
                user_progress['user_progress']['categories'][category]['total_attempted'] = data['total_attempted']
                user_progress['user_progress']['categories'][category]['correct_solutions'] = data['correct_solutions']
        
        # 保存测试场景数据
        scenario_file = f"test_scenarios/{scenario_name}.json"
        with open(scenario_file, 'w') as f:
            json.dump(user_progress, f, indent=4)
        
        print(f"已创建测试场景: {scenario_name}")
        return scenario_file
    
    def apply_test_scenario(self, scenario_file):
        """应用测试场景数据到用户进度文件"""
        if os.path.exists(scenario_file):
            with open(scenario_file, 'r') as f:
                user_progress = json.load(f)
            
            with open('user_progress.json', 'w') as f:
                json.dump(user_progress, f, indent=4)
            
            # 重新初始化推荐系统以加载新的进度数据
            self.recommender = RecommendationSystem()
            print(f"已应用测试场景: {scenario_file}")
        else:
            print(f"测试场景文件不存在: {scenario_file}")
    
    def test_initial_user(self):
        """测试初始用户的推荐效果"""
        print("\n=== 测试场景: 初始用户 ===")
        # 创建一个所有类别都未尝试的场景
        scenario_data = {}
        for category in self.recommender.progress['user_progress']['categories']:
            scenario_data[category] = {'total_attempted': 0, 'correct_solutions': 0}
        
        scenario_file = self.create_test_scenario("initial_user", scenario_data)
        self.apply_test_scenario(scenario_file)
        
        # 获取推荐并分析
        recommendations = self.recommender.get_recommendations(3)
        self.analyze_recommendations(recommendations)
    
    def test_balanced_user(self):
        """测试平衡能力用户的推荐效果"""
        print("\n=== 测试场景: 平衡能力用户 ===")
        # 创建一个在几个类别中有适中表现的场景
        scenario_data = {
            "Arrays and Hashing": {'total_attempted': 5, 'correct_solutions': 3},
            "Two Pointers": {'total_attempted': 4, 'correct_solutions': 2},
            "Stack": {'total_attempted': 3, 'correct_solutions': 2},
            "Binary Search": {'total_attempted': 2, 'correct_solutions': 1}
        }
        
        scenario_file = self.create_test_scenario("balanced_user", scenario_data)
        self.apply_test_scenario(scenario_file)
        
        # 获取推荐并分析
        recommendations = self.recommender.get_recommendations(3)
        self.analyze_recommendations(recommendations)
        
        # 分析Dijkstra算法推荐路径
        self.analyze_dijkstra_path()
    
    def test_advanced_user(self):
        """测试高级用户的推荐效果"""
        print("\n=== 测试场景: 高级用户 ===")
        # 创建一个在大多数类别中都有较高成功率的场景
        scenario_data = {}
        for category in self.recommender.progress['user_progress']['categories']:
            scenario_data[category] = {'total_attempted': 8, 'correct_solutions': 6}
        
        # 添加几个弱项类别
        scenario_data["Dynamic Programming 1D"] = {'total_attempted': 5, 'correct_solutions': 2}
        scenario_data["Graphs"] = {'total_attempted': 4, 'correct_solutions': 1}
        
        scenario_file = self.create_test_scenario("advanced_user", scenario_data)
        self.apply_test_scenario(scenario_file)
        
        # 获取推荐并分析
        recommendations = self.recommender.get_recommendations(3)
        self.analyze_recommendations(recommendations)
        
        # 分析Dijkstra算法推荐路径
        self.analyze_dijkstra_path()
    
    def test_unbalanced_user(self):
        """测试能力不平衡用户的推荐效果"""
        print("\n=== 测试场景: 能力不平衡用户 ===")
        # 创建一个在某些类别表现很好，在某些类别表现很差的场景
        scenario_data = {
            "Arrays and Hashing": {'total_attempted': 10, 'correct_solutions': 9},
            "Two Pointers": {'total_attempted': 8, 'correct_solutions': 7},
            "Dynamic Programming 1D": {'total_attempted': 6, 'correct_solutions': 1},
            "Graphs": {'total_attempted': 5, 'correct_solutions': 0}
        }
        
        scenario_file = self.create_test_scenario("unbalanced_user", scenario_data)
        self.apply_test_scenario(scenario_file)
        
        # 获取推荐并分析
        recommendations = self.recommender.get_recommendations(3)
        self.analyze_recommendations(recommendations)
        
        # 分析Dijkstra算法推荐路径
        self.analyze_dijkstra_path()
    
    def analyze_recommendations(self, recommendations):
        """分析并输出推荐结果"""
        print("\n推荐题目分析:")
        
        # 输出推荐题目
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['name']}")
            print(f"   难度: {rec['difficulty']}")
            print(f"   类别: {rec['category']}")
            print(f"   推荐理由: {rec['reason']}")
        
        # 统计难度分布
        difficulty_counts = {'Easy': 0, 'Medium': 0, 'Hard': 0}
        for rec in recommendations:
            difficulty_counts[rec['difficulty']] += 1
        
        print("\n难度分布:")
        for diff, count in difficulty_counts.items():
            print(f"   {diff}: {count} 题 ({count/len(recommendations)*100:.1f}%)")
        
        # 统计类别分布
        category_counts = {}
        for rec in recommendations:
            category = rec['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print("\n类别分布:")
        for cat, count in category_counts.items():
            print(f"   {cat}: {count} 题")
    
    def analyze_dijkstra_path(self):
        """分析Dijkstra算法的推荐路径"""
        strong_categories = self.recommender.get_user_strong_categories()
        weak_categories = self.recommender.get_user_weak_categories()
        
        print("\nDijkstra算法分析:")
        print(f"强类别: {', '.join(strong_categories) if strong_categories else '无'}")
        print(f"弱类别: {', '.join(weak_categories) if weak_categories else '无'}")
        
        next_category = self.category_graph.get_next_category_via_dijkstra(
            start_categories=strong_categories or list(self.recommender.progress['user_progress']['categories'].keys()),
            target_categories=weak_categories,
            user_progress=self.recommender.progress['user_progress']
        )
        
        print(f"推荐学习路径: {next_category}")
    
    def run_all_tests(self):
        """运行所有测试场景"""
        try:
            self.test_initial_user()
            self.test_balanced_user()
            self.test_advanced_user()
            self.test_unbalanced_user()
        finally:
            # 测试完成后恢复原始用户进度
            self.restore_user_progress()

if __name__ == "__main__":
    tester = RecommenderTester()
    tester.run_all_tests() 