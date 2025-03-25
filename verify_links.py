import json
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def verify_link(problem):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    try:
        response = requests.get(problem['link'], headers=headers, timeout=5)
        if response.status_code == 200:
            return True, problem['name'], problem['link']
        return False, problem['name'], problem['link']
    except:
        return False, problem['name'], problem['link']

def main():
    # 读取问题数据
    with open('problems.json', 'r') as f:
        data = json.load(f)
    
    # 收集所有问题
    all_problems = []
    for category in data.values():
        all_problems.extend(category)
    
    print(f"开始验证 {len(all_problems)} 个链接...")
    
    # 使用线程池并行验证链接
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(tqdm(
            executor.map(verify_link, all_problems),
            total=len(all_problems),
            desc="验证进度"
        ))
    
    # 统计结果
    valid_links = []
    invalid_links = []
    
    for is_valid, name, link in results:
        if is_valid:
            valid_links.append((name, link))
        else:
            invalid_links.append((name, link))
    
    # 打印结果
    print("\n验证结果:")
    print(f"总链接数: {len(all_problems)}")
    print(f"有效链接: {len(valid_links)}")
    print(f"无效链接: {len(invalid_links)}")
    
    if invalid_links:
        print("\n无效链接列表:")
        for name, link in invalid_links:
            print(f"- {name}: {link}")

if __name__ == "__main__":
    main() 