import json
import sys
from pathlib import Path

# 添加父目录到系统路径，以便导入项目模块
sys.path.append(str(Path(__file__).parent.parent))

def check_category_mismatch():
    """检查problems.json和user_progress.json中的类别是否匹配"""
    # 读取problems.json中的类别
    with open('problems.json', 'r') as f:
        problems_data = json.load(f)
    
    problems_categories = set(problems_data.keys())
    print(f"problems.json中的类别 ({len(problems_categories)}):")
    for category in sorted(problems_categories):
        print(f"  - {category}")
    
    # 读取user_progress.json中的类别
    with open('user_progress.json', 'r') as f:
        progress_data = json.load(f)
    
    progress_categories = set(progress_data['user_progress']['categories'].keys())
    print(f"\nuser_progress.json中的类别 ({len(progress_categories)}):")
    for category in sorted(progress_categories):
        print(f"  - {category}")
    
    # 检查不匹配的类别
    problems_only = problems_categories - progress_categories
    progress_only = progress_categories - problems_categories
    
    print("\n=== 分析结果 ===")
    if problems_only:
        print("\n仅在problems.json中存在的类别:")
        for category in sorted(problems_only):
            print(f"  - {category}")
    
    if progress_only:
        print("\n仅在user_progress.json中存在的类别:")
        for category in sorted(progress_only):
            print(f"  - {category}")
    
    if not problems_only and not progress_only:
        print("两个文件中的类别完全匹配！")
    else:
        print("\n可能的问题:")
        if problems_only:
            print(f"- problems.json中有{len(problems_only)}个类别在user_progress.json中不存在")
        if progress_only:
            print(f"- user_progress.json中有{len(progress_only)}个类别在problems.json中不存在")
        
        print("\n解决建议:")
        print("1. 确保两个文件使用完全相同的类别名称")
        print("2. 更新user_progress.json以包含problems.json中的所有类别")
        print("3. 检查类别Graph类的定义是否与这两个文件匹配")

if __name__ == "__main__":
    check_category_mismatch() 