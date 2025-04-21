import json
import os
import sys
from pathlib import Path

# 添加父目录到系统路径，以便导入项目模块
sys.path.append(str(Path(__file__).parent.parent))

def backup_user_progress():
    """备份原始的用户进度文件"""
    if os.path.exists('user_progress.json'):
        with open('user_progress.json', 'r') as f:
            user_progress = json.load(f)
        
        with open('user_progress_backup.json', 'w') as f:
            json.dump(user_progress, f, indent=4)
        print("已备份原始用户进度文件到 user_progress_backup.json")

def restore_user_progress():
    """恢复原始的用户进度文件"""
    if os.path.exists('user_progress_backup.json'):
        with open('user_progress_backup.json', 'r') as f:
            user_progress = json.load(f)
        
        with open('user_progress.json', 'w') as f:
            json.dump(user_progress, f, indent=4)
        print("已恢复原始用户进度文件")

def create_test_profile(profile_name, categories_data):
    """创建测试用户配置文件"""
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
    for category, data in categories_data.items():
        if category in user_progress['user_progress']['categories']:
            user_progress['user_progress']['categories'][category]['total_attempted'] = data['total_attempted']
            user_progress['user_progress']['categories'][category]['correct_solutions'] = data['correct_solutions']
    
    # 保存测试用户配置文件
    os.makedirs('test_profiles', exist_ok=True)
    profile_file = f"test_profiles/{profile_name}.json"
    with open(profile_file, 'w') as f:
        json.dump(user_progress, f, indent=4)
    
    print(f"已创建测试用户配置: {profile_name}")
    return profile_file

def apply_test_profile(profile_file):
    """应用测试用户配置文件到user_progress.json"""
    if os.path.exists(profile_file):
        with open(profile_file, 'r') as f:
            user_progress = json.load(f)
        
        with open('user_progress.json', 'w') as f:
            json.dump(user_progress, f, indent=4)
        
        print(f"已应用测试用户配置: {profile_file}")
    else:
        print(f"测试用户配置文件不存在: {profile_file}")

def generate_user_profiles():
    """生成各种用户配置文件"""
    # 备份原始用户进度
    backup_user_progress()
    
    # 配置文件1: 完全新手用户
    beginner_data = {}  # 所有类别都没有尝试
    create_test_profile("beginner_user", beginner_data)
    
    # 配置文件2: 数组和哈希专家
    array_expert_data = {
        "Arrays and Hashing": {"total_attempted": 10, "correct_solutions": 9},
        "Two Pointers": {"total_attempted": 3, "correct_solutions": 1},
        "Stack": {"total_attempted": 2, "correct_solutions": 1}
    }
    create_test_profile("array_expert", array_expert_data)
    
    # 配置文件3: 有经验但成功率低的用户
    struggling_user_data = {
        "Arrays and Hashing": {"total_attempted": 5, "correct_solutions": 1},
        "Two Pointers": {"total_attempted": 4, "correct_solutions": 1},
        "Stack": {"total_attempted": 3, "correct_solutions": 0},
        "Binary Search": {"total_attempted": 2, "correct_solutions": 0}
    }
    create_test_profile("struggling_user", struggling_user_data)
    
    # 配置文件4: 有经验且平衡发展的用户
    balanced_user_data = {
        "Arrays and Hashing": {"total_attempted": 6, "correct_solutions": 4},
        "Two Pointers": {"total_attempted": 5, "correct_solutions": 3},
        "Stack": {"total_attempted": 4, "correct_solutions": 3},
        "Binary Search": {"total_attempted": 3, "correct_solutions": 2},
        "Sliding Window": {"total_attempted": 3, "correct_solutions": 2}
    }
    create_test_profile("balanced_user", balanced_user_data)
    
    # 配置文件5: 高级用户，已尝试多个类别
    advanced_user_data = {
        "Arrays and Hashing": {"total_attempted": 10, "correct_solutions": 8},
        "Two Pointers": {"total_attempted": 8, "correct_solutions": 7},
        "Stack": {"total_attempted": 6, "correct_solutions": 5},
        "Binary Search": {"total_attempted": 7, "correct_solutions": 6},
        "Sliding Window": {"total_attempted": 5, "correct_solutions": 4},
        "Linked List": {"total_attempted": 4, "correct_solutions": 3},
        "Trees": {"total_attempted": 4, "correct_solutions": 3},
        "Dynamic Programming 1D": {"total_attempted": 3, "correct_solutions": 2}
    }
    create_test_profile("advanced_user", advanced_user_data)
    
    # 配置文件6: 动态规划弱点用户
    dp_weakness_data = {
        "Arrays and Hashing": {"total_attempted": 8, "correct_solutions": 7},
        "Two Pointers": {"total_attempted": 7, "correct_solutions": 6},
        "Stack": {"total_attempted": 6, "correct_solutions": 5},
        "Dynamic Programming 1D": {"total_attempted": 5, "correct_solutions": 1},
        "Dynamic Programming 2D": {"total_attempted": 3, "correct_solutions": 0}
    }
    create_test_profile("dp_weakness_user", dp_weakness_data)
    
    # 配置文件7: 图算法弱点用户
    graph_weakness_data = {
        "Arrays and Hashing": {"total_attempted": 9, "correct_solutions": 8},
        "Two Pointers": {"total_attempted": 8, "correct_solutions": 7},
        "Stack": {"total_attempted": 7, "correct_solutions": 6},
        "Graphs": {"total_attempted": 4, "correct_solutions": 1},
        "Trees": {"total_attempted": 5, "correct_solutions": 2}
    }
    create_test_profile("graph_weakness_user", graph_weakness_data)
    
    # 配置文件8: 不平衡用户，有强项和弱项
    unbalanced_user_data = {
        "Arrays and Hashing": {"total_attempted": 10, "correct_solutions": 9},
        "Two Pointers": {"total_attempted": 9, "correct_solutions": 8},
        "Stack": {"total_attempted": 8, "correct_solutions": 7},
        "Binary Search": {"total_attempted": 7, "correct_solutions": 6},
        "Dynamic Programming 1D": {"total_attempted": 6, "correct_solutions": 1},
        "Graphs": {"total_attempted": 5, "correct_solutions": 0},
        "Heap/Priority Queue": {"total_attempted": 4, "correct_solutions": 1}
    }
    create_test_profile("unbalanced_user", unbalanced_user_data)
    
    # 恢复原始用户进度
    restore_user_progress()
    
    # 打印生成的所有配置文件
    print("\n生成的测试用户配置文件:")
    for file in os.listdir('test_profiles'):
        if file.endswith('.json'):
            print(f"- {file}")

def test_recommendation(profile_name):
    """加载指定的用户配置并获取推荐"""
    from recommendation_system import RecommendationSystem
    
    profile_file = f"test_profiles/{profile_name}.json"
    if not os.path.exists(profile_file):
        print(f"测试用户配置文件不存在: {profile_file}")
        return
    
    # 备份当前用户进度
    backup_user_progress()
    
    try:
        # 应用测试用户配置
        apply_test_profile(profile_file)
        
        # 初始化推荐系统
        recommender = RecommendationSystem()
        
        # 获取并打印强弱类别
        strong_categories = recommender.get_user_strong_categories()
        weak_categories = recommender.get_user_weak_categories()
        print(f"\n用户 '{profile_name}' 的状态:")
        print(f"强类别: {', '.join(strong_categories) if strong_categories else '无'}")
        print(f"弱类别: {', '.join(weak_categories) if weak_categories else '无'}")
        
        # 获取并打印推荐
        recommendations = recommender.get_recommendations(3)
        print("\n推荐问题:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['name']}")
            print(f"   难度: {rec['difficulty']}")
            print(f"   类别: {rec['category']}")
            print(f"   推荐理由: {rec['reason']}")
        
    finally:
        # 恢复原始用户进度
        restore_user_progress()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 如果提供了参数，测试指定的用户配置
        test_recommendation(sys.argv[1])
    else:
        # 没有参数，生成所有用户配置
        generate_user_profiles()
        print("\n使用方法: python test_user_profiles.py <profile_name>")
        print("例如: python test_user_profiles.py beginner_user") 