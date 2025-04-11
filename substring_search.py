def build_kmp_table(pattern: str) -> list:
    """build the partial match table for KMP algorithm"""
    table = [0] * len(pattern)
    j = 0
    
    for i in range(1, len(pattern)):
        while j > 0 and pattern[i] != pattern[j]:
            j = table[j - 1]
            
        if pattern[i] == pattern[j]:
            j += 1
            
        table[i] = j
        
    return table

def kmp_search(text: str, pattern: str) -> bool:
    """KMP search algorithm"""
    if not pattern:
        return True
        
    if not text:
        return False
        
    table = build_kmp_table(pattern.lower())
    j = 0
    
    text = text.lower()
    
    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = table[j - 1]
            
        if text[i] == pattern[j]:
            j += 1
            
        if j == len(pattern):
            return True
            
    return False

def search_problems(keyword: str) -> list:
    """search for problems containing the keyword"""
    import json
    
    with open('problems.json', 'r') as f:
        problems = json.load(f)
    
    results = []
    
    for category, problem_list in problems.items():
        for problem in problem_list:
            # search in the problem name
            if kmp_search(problem['name'], keyword):
                results.append(problem)
                continue
                
            # search in the category name
            if kmp_search(problem['category'], keyword):
                results.append(problem)
                continue

    return results
