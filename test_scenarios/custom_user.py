import json
import os
import sys
from pathlib import Path

# 添加父目录到系统路径，以便导入项目模块
sys.path.append(str(Path(__file__).parent.parent))

from recommendation_system import RecommendationSystem
from category_graph import CategoryGraph

def create_custom_user(all_categories_explored=False):
    """
    创建一个定制的用户配置文件
    
    Args:
        all_categories_explored: 是否已经探索了所有类别
    """
    # 读取原始用户进度文件作为模板
    with open('user_progress.json', 'r') as f:
        user_progress = json.load(f)
    
    # 清空所有类别的尝试次数和正确解决次数
    for category in user_progress['user_progress']['categories']:
        user_progress['user_progress']['categories'][category] = {
            'total_attempted': 0,
            'correct_solutions': 0
        }
    
    if all_categories_explored:
        # 创建一个已探索所有类别的用户
        # 设置一些强项类别（高成功率）
        strong_categories = {
            'Arrays and Hashing': {'total_attempted': 12, 'correct_solutions': 10},
            'Two Pointers': {'total_attempted': 10, 'correct_solutions': 8},
            'Stack': {'total_attempted': 8, 'correct_solutions': 7}
        }
        
        # 设置一些一般水平的类别（中等成功率）
        medium_categories = {
            'Sliding Window': {'total_attempted': 6, 'correct_solutions': 4},
            'Binary Search': {'total_attempted': 5, 'correct_solutions': 3},
            'Linked List': {'total_attempted': 4, 'correct_solutions': 2},
            'Tries': {'total_attempted': 3, 'correct_solutions': 2}
        }
        
        # 设置一些弱项类别（低成功率）
        weak_categories = {
            'Dynamic Programming 1D': {'total_attempted': 5, 'correct_solutions': 1},
            'Dynamic Programming 2D': {'total_attempted': 3, 'correct_solutions': 0},
            'Graphs': {'total_attempted': 4, 'correct_solutions': 1}
        }
        
        # 确保所有其他类别至少有一次尝试
        other_categories = {
            'Trees': {'total_attempted': 2, 'correct_solutions': 1},
            'Heap/Priority Queue': {'total_attempted': 2, 'correct_solutions': 1},
            'Backtracking': {'total_attempted': 2, 'correct_solutions': 1},
            'Greedy': {'total_attempted': 2, 'correct_solutions': 1}
        }
        
        # 更新类别数据
        for category, data in {**strong_categories, **medium_categories, **weak_categories, **other_categories}.items():
            if category in user_progress['user_progress']['categories']:
                user_progress['user_progress']['categories'][category]['total_attempted'] = data['total_attempted']
                user_progress['user_progress']['categories'][category]['correct_solutions'] = data['correct_solutions']
        
        # 添加一些已解决的问题
        solved_problems = [
            'Two Sum', 'Valid Anagram', 'Contains Duplicate',  # Arrays and Hashing
            'Valid Palindrome', 'Two Sum II',  # Two Pointers
            'Valid Parentheses', 'Min Stack',  # Stack
            'Binary Search', 'Search a 2D Matrix',  # Binary Search
            'Climbing Stairs',  # Dynamic Programming 1D
            'Number of Islands'  # Graphs
        ]
        
        # 添加一些尝试但未解决的问题
        unsolved_problems = [
            'Longest Consecutive Sequence',  # Arrays and Hashing 
            'Trapping Rain Water',  # Two Pointers
            'Longest Substring Without Repeating Characters',  # Sliding Window
            'Unique Paths',  # Dynamic Programming 1D
            'Best Time to Buy and Sell Stock',  # Greedy/Sliding Window
            'Pacific Atlantic Water Flow'  # Graphs
        ]
        
        # 更新问题记录
        user_progress['user_progress']['problems'] = {}
        
        for problem in solved_problems:
            user_progress['user_progress']['problems'][problem] = {
                'attempts': 1,
                'solved': True,
                'solutions': [
                    {
                        'file_path': f"solutions/{problem.replace(' ', '_')}.py",
                        'score': 0.8,
                        'feedback': "Simulated feedback for solved problem"
                    }
                ]
            }
        
        for problem in unsolved_problems:
            user_progress['user_progress']['problems'][problem] = {
                'attempts': 2,
                'solved': False,
                'solutions': [
                    {
                        'file_path': f"solutions/{problem.replace(' ', '_')}.py",
                        'score': 0.4,
                        'feedback': "Simulated feedback for unsolved problem"
                    }
                ]
            }
    else:
        # 创建一个只探索了部分类别的用户
        explored_categories = {
            'Arrays and Hashing': {'total_attempted': 8, 'correct_solutions': 6},
            'Two Pointers': {'total_attempted': 6, 'correct_solutions': 4},
            'Stack': {'total_attempted': 4, 'correct_solutions': 3},
            'Binary Search': {'total_attempted': 3, 'correct_solutions': 2}
        }
        
        # 更新类别数据
        for category, data in explored_categories.items():
            if category in user_progress['user_progress']['categories']:
                user_progress['user_progress']['categories'][category]['total_attempted'] = data['total_attempted']
                user_progress['user_progress']['categories'][category]['correct_solutions'] = data['correct_solutions']
        
        # 添加一些已解决的问题
        solved_problems = [
            'Two Sum', 'Valid Anagram',  # Arrays and Hashing
            'Valid Palindrome',  # Two Pointers
            'Valid Parentheses',  # Stack
            'Binary Search'  # Binary Search
        ]
        
        # 添加一些尝试但未解决的问题
        unsolved_problems = [
            'Group Anagrams',  # Arrays and Hashing
            '3Sum',  # Two Pointers
            'Min Stack',  # Stack
            'Search a 2D Matrix'  # Binary Search
        ]
        
        # 更新问题记录
        user_progress['user_progress']['problems'] = {}
        
        for problem in solved_problems:
            user_progress['user_progress']['problems'][problem] = {
                'attempts': 1,
                'solved': True,
                'solutions': [
                    {
                        'file_path': f"solutions/{problem.replace(' ', '_')}.py",
                        'score': 0.8,
                        'feedback': "Simulated feedback for solved problem"
                    }
                ]
            }
        
        for problem in unsolved_problems:
            user_progress['user_progress']['problems'][problem] = {
                'attempts': 2,
                'solved': False,
                'solutions': [
                    {
                        'file_path': f"solutions/{problem.replace(' ', '_')}.py",
                        'score': 0.4,
                        'feedback': "Simulated feedback for unsolved problem"
                    }
                ]
            }
    
    # 保存用户配置文件
    profile_type = "fully_explored" if all_categories_explored else "partially_explored"
    profile_file = f"test_profiles/custom_{profile_type}_user.json"
    
    os.makedirs('test_profiles', exist_ok=True)
    with open(profile_file, 'w') as f:
        json.dump(user_progress, f, indent=4)
    
    print(f"已创建定制用户配置: {profile_file}")
    return profile_file

def test_custom_user(profile_file):
    """测试定制用户的推荐效果"""
    # 备份当前用户进度
    if os.path.exists('user_progress.json'):
        with open('user_progress.json', 'r') as f:
            orig_progress = json.load(f)
    
    try:
        # 应用定制用户配置
        with open(profile_file, 'r') as f:
            user_progress = json.load(f)
        
        with open('user_progress.json', 'w') as f:
            json.dump(user_progress, f, indent=4)
        
        # 初始化推荐系统和类别图
        recommender = RecommendationSystem()
        category_graph = CategoryGraph()
        
        # 获取用户强弱类别
        strong_categories = recommender.get_user_strong_categories()
        weak_categories = recommender.get_user_weak_categories()
        
        profile_name = os.path.basename(profile_file).split('.')[0]
        print(f"\n=== 定制用户 '{profile_name}' 分析 ===")
        print(f"强类别: {', '.join(strong_categories) if strong_categories else '无'}")
        print(f"弱类别: {', '.join(weak_categories) if weak_categories else '无'}")
        
        # 检查是否是初始用户
        is_initial = recommender.is_initial_user()
        print(f"是否初始用户: {is_initial}")
        
        # 检查已尝试的类别数
        attempted_categories = sum(1 for cat, stats in recommender.progress['user_progress']['categories'].items() 
                                 if stats['total_attempted'] > 0)
        total_categories = len(recommender.progress['user_progress']['categories'])
        print(f"已尝试类别数: {attempted_categories}/{total_categories}")
        
        # 获取Dijkstra推荐的下一个类别
        next_category = category_graph.get_next_category_via_dijkstra(
            start_categories=strong_categories or list(recommender.progress['user_progress']['categories'].keys()),
            target_categories=weak_categories or [],
            user_progress=recommender.progress['user_progress']
        )
        
        print(f"Dijkstra推荐的下一个学习类别: {next_category}")
        
        # 获取推荐问题
        recommendations = recommender.get_recommendations(5)
        
        print("\n推荐问题:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['name']}")
            print(f"   难度: {rec['difficulty']}")
            print(f"   类别: {rec['category']}")
            print(f"   推荐理由: {rec['reason']}")
        
        # 统计推荐的类别分布
        category_counts = {}
        for rec in recommendations:
            category = rec['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print("\n推荐类别分布:")
        for category, count in category_counts.items():
            percentage = count / len(recommendations) * 100
            print(f"  {category}: {count} 题 ({percentage:.1f}%)")
        
        # 检查推荐中是否包含弱项类别
        contains_weak_category = any(rec['category'] in weak_categories for rec in recommendations)
        contains_dijkstra_category = any(rec['category'] == next_category for rec in recommendations)
        
        print("\n推荐质量评估:")
        if contains_weak_category:
            print("✓ 推荐中包含用户弱项类别")
        else:
            print("✗ 推荐中不包含用户弱项类别")
        
        if contains_dijkstra_category:
            print(f"✓ 推荐中包含Dijkstra建议的类别 '{next_category}'")
        elif next_category:
            print(f"✗ 推荐中不包含Dijkstra建议的类别 '{next_category}'")
        
        # 评估整体推荐质量
        difficulty_counts = {'Easy': 0, 'Medium': 0, 'Hard': 0}
        for rec in recommendations:
            difficulty_counts[rec['difficulty']] += 1
        
        print("\n难度分布:")
        for diff, count in difficulty_counts.items():
            percentage = count / len(recommendations) * 100
            print(f"  {diff}: {count} 题 ({percentage:.1f}%)")
        
    finally:
        # 恢复原始用户进度
        if 'orig_progress' in locals():
            with open('user_progress.json', 'w') as f:
                json.dump(orig_progress, f, indent=4)

if __name__ == "__main__":
    # 创建并测试完全探索的用户
    fully_explored_file = create_custom_user(all_categories_explored=True)
    test_custom_user(fully_explored_file)
    
    # 创建并测试部分探索的用户
    partially_explored_file = create_custom_user(all_categories_explored=False)
    test_custom_user(partially_explored_file) 