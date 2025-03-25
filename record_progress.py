import json
import time
from datetime import datetime

def load_data():
    with open('problems.json', 'r') as f:
        problems = json.load(f)
    with open('user_progress.json', 'r') as f:
        progress = json.load(f)
    return problems, progress

def save_progress(progress):
    with open('user_progress.json', 'w') as f:
        json.dump(progress, f, indent=4)

def record_problem_attempt(problem_name, category, is_correct, time_taken):
    problems, progress = load_data()
    
    # 记录具体题目
    if problem_name not in progress['user_progress']['problems']:
        progress['user_progress']['problems'][problem_name] = {
            'attempts': 0,
            'correct_solutions': 0,
            'total_time': 0,
            'last_attempted': None
        }
    
    problem_progress = progress['user_progress']['problems'][problem_name]
    problem_progress['attempts'] += 1
    if is_correct:
        problem_progress['correct_solutions'] += 1
    problem_progress['total_time'] += time_taken
    problem_progress['last_attempted'] = datetime.now().isoformat()
    
    # 更新分类统计
    category_progress = progress['user_progress']['categories'][category]
    category_progress['total_attempted'] += 1
    if is_correct:
        category_progress['correct_solutions'] += 1
    category_progress['average_time'] = (
        (category_progress['average_time'] * (category_progress['total_attempted'] - 1) + time_taken)
        / category_progress['total_attempted']
    )
    
    save_progress(progress)
    print(f"已记录 {problem_name} 的做题记录")

def show_progress_summary():
    problems, progress = load_data()
    
    print("\n=== 做题进度总结 ===")
    print("\n按分类统计:")
    for category, stats in progress['user_progress']['categories'].items():
        if stats['total_attempted'] > 0:
            success_rate = (stats['correct_solutions'] / stats['total_attempted']) * 100
            print(f"\n{category}:")
            print(f"  尝试次数: {stats['total_attempted']}")
            print(f"  正确率: {success_rate:.1f}%")
            print(f"  平均用时: {stats['average_time']:.1f}秒")
    
    print("\n最近做题记录:")
    recent_problems = sorted(
        progress['user_progress']['problems'].items(),
        key=lambda x: x[1]['last_attempted'] if x[1]['last_attempted'] else '',
        reverse=True
    )[:5]
    
    for problem, stats in recent_problems:
        if stats['attempts'] > 0:
            success_rate = (stats['correct_solutions'] / stats['attempts']) * 100
            print(f"\n{problem}:")
            print(f"  尝试次数: {stats['attempts']}")
            print(f"  正确率: {success_rate:.1f}%")
            print(f"  平均用时: {stats['total_time']/stats['attempts']:.1f}秒")
            print(f"  最后尝试: {stats['last_attempted']}")

def main():
    while True:
        print("\n=== LeetCode 做题记录系统 ===")
        print("1. 记录做题")
        print("2. 查看进度")
        print("3. 退出")
        
        choice = input("\n请选择操作 (1-3): ")
        
        if choice == '1':
            problem_name = input("请输入题目名称: ")
            category = input("请输入题目分类: ")
            is_correct = input("是否做对? (y/n): ").lower() == 'y'
            time_taken = float(input("用时(秒): "))
            
            record_problem_attempt(problem_name, category, is_correct, time_taken)
        
        elif choice == '2':
            show_progress_summary()
        
        elif choice == '3':
            break
        
        else:
            print("无效的选择，请重试")

if __name__ == "__main__":
    main() 