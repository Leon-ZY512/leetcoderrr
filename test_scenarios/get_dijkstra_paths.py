import json
import os
import sys
from pathlib import Path

# 添加父目录到系统路径，以便导入项目模块
sys.path.append(str(Path(__file__).parent.parent))

from recommendation_system import RecommendationSystem
from category_graph import CategoryGraph

def analyze_dijkstra_path(profile_name):
    """分析指定用户配置的Dijkstra推荐路径"""
    profile_file = f"test_profiles/{profile_name}.json"
    if not os.path.exists(profile_file):
        print(f"测试用户配置文件不存在: {profile_file}")
        return
    
    # 备份当前用户进度
    if os.path.exists('user_progress.json'):
        with open('user_progress.json', 'r') as f:
            orig_progress = json.load(f)
    
    try:
        # 应用测试用户配置
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
        
        print(f"\n=== 用户 '{profile_name}' 的Dijkstra路径分析 ===")
        print(f"强类别: {', '.join(strong_categories) if strong_categories else '无'}")
        print(f"弱类别: {', '.join(weak_categories) if weak_categories else '无'}")
        
        # 获取下一个推荐类别
        next_category = category_graph.get_next_category_via_dijkstra(
            start_categories=strong_categories or list(recommender.progress['user_progress']['categories'].keys()),
            target_categories=weak_categories or [],
            user_progress=recommender.progress['user_progress']
        )
        
        print(f"Dijkstra推荐的下一个学习类别: {next_category}")
        
        # 获取并分析推荐
        recommendations = recommender.get_recommendations(3)
        
        # 检查推荐中是否包含Dijkstra推荐的类别
        contains_dijkstra_category = any(rec['category'] == next_category for rec in recommendations)
        
        print("\n推荐问题类别:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['category']} ({rec['difficulty']})")
        
        if contains_dijkstra_category:
            print(f"\n结论: 推荐中包含Dijkstra建议的类别 '{next_category}'")
        else:
            print(f"\n结论: 推荐中不包含Dijkstra建议的类别 '{next_category}'")
            print("可能原因:")
            print("1. 系统优先推荐未尝试过的类别")
            print("2. 用户尚未完成所有类别的探索")
        
    finally:
        # 恢复原始用户进度
        if 'orig_progress' in locals():
            with open('user_progress.json', 'w') as f:
                json.dump(orig_progress, f, indent=4)

def analyze_all_profiles():
    """分析所有测试用户配置的Dijkstra推荐路径"""
    if not os.path.exists('test_profiles'):
        print("test_profiles目录不存在，请先运行test_user_profiles.py生成测试用户配置")
        return
    
    for file in os.listdir('test_profiles'):
        if file.endswith('.json'):
            profile_name = file.split('.')[0]
            analyze_dijkstra_path(profile_name)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 如果提供了参数，分析指定的用户配置
        analyze_dijkstra_path(sys.argv[1])
    else:
        # 没有参数，分析所有用户配置
        analyze_all_profiles() 