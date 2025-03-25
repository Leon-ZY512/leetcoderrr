import json
import random
from datetime import datetime
from typing import Dict, List
import openai

class CodeEvaluator:
    def __init__(self):
        with open('user_progress.json', 'r') as f:
            self.progress = json.load(f)
        # 设置 OpenAI API key
        openai.api_key = "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"
    
    def evaluate_code(self, problem_name: str, code: str, time_taken: float) -> Dict:
        """使用 GPT API 评估用户提交的代码"""
        try:
            # 构建评估提示
            prompt = f"""请评估以下 LeetCode 题目 '{problem_name}' 的代码，从以下五个方面进行评分（0-1分）：
1. 正确性：代码是否能正确处理所有测试用例
2. 时间复杂度：算法的时间复杂度是否最优
3. 空间复杂度：算法的空间复杂度是否最优
4. 代码风格：是否符合编程规范
5. 可读性：代码是否清晰易懂

代码：
```python
{code}
```

请以 JSON 格式返回评分和详细反馈，格式如下：
{{
    "scores": {{
        "correctness": 0.8,
        "time_complexity": 0.7,
        "space_complexity": 0.9,
        "code_style": 0.8,
        "readability": 0.7
    }},
    "feedback": [
        "具体改进建议1",
        "具体改进建议2"
    ]
}}"""

            # 调用 GPT API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一个专业的代码审查员，擅长评估算法代码的质量。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # 解析响应
            result = json.loads(response.choices[0].message.content)
            
            # 更新进度
            self._update_progress(problem_name, result['scores'], time_taken)
            
            return result
            
        except Exception as e:
            print(f"评估过程中出现错误: {str(e)}")
            # 如果 API 调用失败，使用模拟评分
            return self._fallback_evaluation(code, time_taken)
    
    def _fallback_evaluation(self, code: str, time_taken: float) -> Dict:
        """当 API 调用失败时的备用评估方法"""
        scores = {
            'correctness': random.uniform(0.6, 1.0),
            'time_complexity': random.uniform(0.6, 1.0),
            'space_complexity': random.uniform(0.6, 1.0),
            'code_style': random.uniform(0.6, 1.0),
            'readability': random.uniform(0.6, 1.0)
        }
        
        total_score = sum(scores.values()) / len(scores)
        
        return {
            'scores': scores,
            'total_score': total_score,
            'feedback': self._generate_feedback(scores)
        }
    
    def _update_progress(self, problem_name: str, scores: Dict, time_taken: float):
        """更新用户进度"""
        if problem_name not in self.progress['user_progress']['problems']:
            self.progress['user_progress']['problems'][problem_name] = {
                'attempts': 0,
                'correct_solutions': 0,
                'total_time': 0,
                'last_attempted': None,
                'scores': []
            }
        
        problem_progress = self.progress['user_progress']['problems'][problem_name]
        problem_progress['attempts'] += 1
        if scores['correctness'] >= 0.8:  # 正确性达到80%以上算作正确
            problem_progress['correct_solutions'] += 1
        problem_progress['total_time'] += time_taken
        problem_progress['last_attempted'] = datetime.now().isoformat()
        problem_progress['scores'].append(scores)
        
        # 更新分类统计
        category = self._get_problem_category(problem_name)
        if category:
            category_progress = self.progress['user_progress']['categories'][category]
            category_progress['total_attempted'] += 1
            if scores['correctness'] >= 0.8:
                category_progress['correct_solutions'] += 1
            category_progress['average_time'] = (
                (category_progress['average_time'] * (category_progress['total_attempted'] - 1) + time_taken)
                / category_progress['total_attempted']
            )
        
        # 保存更新后的进度
        with open('user_progress.json', 'w') as f:
            json.dump(self.progress, f, indent=4)
    
    def _get_problem_category(self, problem_name: str) -> str:
        """获取题目所属分类"""
        with open('problems.json', 'r') as f:
            problems = json.load(f)
            for category, problem_list in problems.items():
                for problem in problem_list:
                    if problem['name'] == problem_name:
                        return category
        return None
    
    def _generate_feedback(self, scores: Dict) -> List[str]:
        """生成反馈建议"""
        feedback = []
        if scores['correctness'] < 0.8:
            feedback.append("代码正确性有待提高，建议仔细检查边界条件和特殊情况")
        if scores['time_complexity'] < 0.8:
            feedback.append("时间复杂度可以优化，考虑使用更高效的算法")
        if scores['space_complexity'] < 0.8:
            feedback.append("空间复杂度可以优化，考虑使用原地操作或更节省空间的数据结构")
        if scores['code_style'] < 0.8:
            feedback.append("代码风格可以改进，建议遵循编程规范")
        if scores['readability'] < 0.8:
            feedback.append("代码可读性可以提升，建议添加适当的注释和文档")
        return feedback

def main():
    evaluator = CodeEvaluator()
    
    while True:
        print("\n=== LeetCode 代码评估系统 ===")
        print("1. 提交代码评估")
        print("2. 查看历史评估")
        print("3. 退出")
        
        choice = input("\n请选择操作 (1-3): ")
        
        if choice == '1':
            problem_name = input("请输入题目名称: ")
            code = input("请输入你的代码: ")
            time_taken = float(input("用时(秒): "))
            
            result = evaluator.evaluate_code(problem_name, code, time_taken)
            
            print("\n=== 评估结果 ===")
            print(f"总分: {result['total_score']:.2f}")
            print("\n详细评分:")
            for metric, score in result['scores'].items():
                print(f"{metric}: {score:.2f}")
            
            print("\n反馈建议:")
            for feedback in result['feedback']:
                print(f"- {feedback}")
        
        elif choice == '2':
            # 显示最近的评估记录
            print("\n=== 最近的评估记录 ===")
            recent_problems = sorted(
                evaluator.progress['user_progress']['problems'].items(),
                key=lambda x: x[1]['last_attempted'] if x[1]['last_attempted'] else '',
                reverse=True
            )[:5]
            
            for problem, stats in recent_problems:
                if stats['attempts'] > 0:
                    print(f"\n{problem}:")
                    print(f"  尝试次数: {stats['attempts']}")
                    print(f"  正确率: {(stats['correct_solutions']/stats['attempts'])*100:.1f}%")
                    print(f"  平均用时: {stats['total_time']/stats['attempts']:.1f}秒")
                    if stats['scores']:
                        latest_score = stats['scores'][-1]
                        print(f"  最新评分: {sum(latest_score.values())/len(latest_score):.2f}")
        
        elif choice == '3':
            break
        
        else:
            print("无效的选择，请重试")

if __name__ == "__main__":
    main() 