import json
import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from collections import defaultdict
import heapq
from category_graph import CategoryGraph

def calculate_category_metrics():
    """Calculate user performance metrics for each category"""
    with open('user_progress.json', 'r') as f:
        user_data = json.load(f)
    
    categories_metrics = {}
    for category, stats in user_data['user_progress']['categories'].items():
        total = stats['total_attempted']
        correct = stats['correct_solutions']
        
        if total > 0:
            accuracy = correct / total
        else:
            accuracy = 0
        
        categories_metrics[category] = {
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            'user_data': stats
        }
    
    # Read completed problems
    solved_problems = {}
    for problem_name, problem_data in user_data['user_progress'].get('problems', {}).items():
        if problem_data.get('solutions', []):
            latest_solution = problem_data['solutions'][-1]
            score = latest_solution.get('score', 0.0)
            solved_problems[problem_name] = {
                'score': score,
                'attempts': problem_data.get('attempts', 1),
                'solved': problem_data.get('solved', False)
            }
    
    # Determine strong and weak categories
    strong_categories = []
    weak_categories = []
    
    # Sort by success rate
    sorted_categories = sorted(
        [(cat, metrics['accuracy']) for cat, metrics in categories_metrics.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # According to the actual logic: categories with success rate above 80% are strong, take the top 2
    for cat, acc in sorted_categories:
        if acc >= 0.8 and len(strong_categories) < 2:
            strong_categories.append(cat)
    
    # If no categories with success rate above 80%, choose the top 2
    if not strong_categories and len(sorted_categories) >= 2:
        strong_categories = [cat for cat, _ in sorted_categories[:2]]
    
    # According to the actual logic: categories with success rate below 60% are weak, take the top 3
    for cat, acc in sorted_categories[::-1]:
        if acc < 0.6 and len(weak_categories) < 3:
            weak_categories.append(cat)
    
    # If not enough weak categories, choose the lowest
    if len(weak_categories) < 3 and len(sorted_categories) >= 3:
        remaining = [cat for cat, _ in sorted_categories[::-1] 
                   if cat not in weak_categories and cat not in strong_categories]
        weak_categories.extend(remaining[:3-len(weak_categories)])
    
    return categories_metrics, solved_problems, strong_categories, weak_categories, user_data['user_progress']

def load_graph_structure():
    """Load predefined category relationship graph from CategoryGraph class"""
    # Create an instance of CategoryGraph to access its graph
    category_graph = CategoryGraph()
    
    # Return the graph structure from CategoryGraph
    return category_graph.graph

def get_problem_categories():
    """Parse the mapping from problem names to categories"""
    problem_categories = {
        "Contains Duplicate": "Arrays and Hashing",
        "Valid Anagram": "Arrays and Hashing",
        "Two Sum": "Arrays and Hashing",
        "Group Anagrams": "Arrays and Hashing",
        "Valid Palindrome": "Two Pointers",
        "3Sum": "Two Pointers",
        "Clone Graph": "Graphs",
        "Course Schedule": "Graphs",
        "Number of Islands": "Graphs",
        "Pacific Atlantic Water Flow": "Graphs",
        "Climbing Stairs": "Dynamic Programming 1D",
        "Coin Change": "Dynamic Programming 1D",
        "Longest Common Subsequence": "Dynamic Programming 2D"
    }
    return problem_categories

def compute_weight(category, metrics, solved_problems, problem_categories):
    """Calculate the dynamic weight for a category based on actual implementation"""
    # Get category data
    data = metrics.get(category, {
        'accuracy': 0,
        'total': 0,
        'correct': 0
    })
    
    # Calculate accuracy
    attempted = data['total']
    correct = data['correct']
    accuracy = correct / attempted if attempted > 0 else 0
    
    # Assume time factor is 0 (not provided in the data)
    normalized_time = 0
    
    # Calculate average problem score for this category
    avg_problem_score = 0.0
    problem_scores_count = 0
    
    for problem_name, problem_data in solved_problems.items():
        if problem_categories.get(problem_name) == category:
            avg_problem_score += problem_data['score']
            problem_scores_count += 1
    
    if problem_scores_count > 0:
        avg_problem_score /= problem_scores_count
    else:
        avg_problem_score = 0.5  # Default value
    
    # Weight calculation formula (same as in category_graph.py)
    weight = (1 - accuracy) * 0.4 + (1 - avg_problem_score) * 0.3 + normalized_time * 0.3
    
    return {
        'accuracy': accuracy,
        'avg_problem_score': avg_problem_score,
        'normalized_time': normalized_time,
        'weight': weight,
        'component1': (1 - accuracy) * 0.4,
        'component2': (1 - avg_problem_score) * 0.3,
        'component3': normalized_time * 0.3
    }

def simulate_dijkstra(graph, metrics, solved_problems, problem_categories, strong_categories, weak_categories):
    """Simulate Dijkstra algorithm flow using the same algorithm as the actual implementation"""
    # Initialize data structures
    distances = {category: float('infinity') for category in graph}
    previous = {category: None for category in graph}
    visited = set()
    
    # Record each step in the process
    steps = []
    
    # Initialize distances for starting points
    for start in strong_categories:
        distances[start] = 0
    
    # Priority queue (use heapq to match the actual implementation)
    queue = [(0, start) for start in strong_categories]
    heapq.heapify(queue)
    
    while queue:
        # Get node with smallest distance
        current_distance, current_node = heapq.heappop(queue)
        
        # Record current step
        current_step = {
            'current_node': current_node,
            'current_distance': current_distance,
            'distances': distances.copy(),
            'previous': previous.copy(),
            'queue': queue.copy(),
            'edges_considered': []
        }
        
        # If we've reached a target, build path and return
        if current_node in weak_categories:
            path = []
            node = current_node
            while node:
                path.append(node)
                node = previous[node]
            path.reverse()
            
            # Record that we found a target
            current_step['reached_target'] = True
            current_step['target'] = current_node
            current_step['path'] = path
            steps.append(current_step)
            
            return {
                'steps': steps,
                'path': path,
                'distance': current_distance,
                'distances': distances,
                'previous': previous
            }
        
        # If already visited, skip
        if current_node in visited:
            continue
        
        # Mark as visited
        visited.add(current_node)
        
        # Check neighbors
        for neighbor, base_weight in graph[current_node].items():
            if neighbor in visited:
                continue
            
            # Calculate dynamic weight
            weight_data = compute_weight(neighbor, metrics, solved_problems, problem_categories)
            dynamic_weight_component = weight_data['weight']
            adjusted_weight = dynamic_weight_component * base_weight
            
            # Record edge being considered
            edge_info = {
                'from': current_node,
                'to': neighbor,
                'base_weight': base_weight,
                'dynamic_component': dynamic_weight_component,
                'adjusted_weight': adjusted_weight,
                'accuracy': weight_data['accuracy'],
                'avg_problem_score': weight_data['avg_problem_score'],
                'weight_calculation': {
                    'component1': weight_data['component1'],
                    'component2': weight_data['component2'],
                    'component3': weight_data['component3']
                },
                'new_distance': current_distance + adjusted_weight,
                'updated': False
            }
            
            # Calculate new distance
            distance = current_distance + adjusted_weight
            
            # If we found a shorter path
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))
                edge_info['updated'] = True
            
            current_step['edges_considered'].append(edge_info)
        
        steps.append(current_step)
    
    # If we reach here, no path was found to any target
    return {
        'steps': steps,
        'path': None,
        'distance': float('infinity'),
        'distances': distances,
        'previous': previous
    }

def visualize_dijkstra_process(graph, metrics, solved_problems, problem_categories, strong_categories, weak_categories, dijkstra_result):
    """Visualize the Dijkstra algorithm process"""
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges
    for source, targets in graph.items():
        for target, weight in targets.items():
            # Calculate dynamic weight
            weight_data = compute_weight(target, metrics, solved_problems, problem_categories)
            dynamic_component = weight_data['weight']
            adjusted_weight = dynamic_component * weight
            G.add_edge(source, target, base_weight=weight, 
                      dynamic_component=dynamic_component,
                      adjusted_weight=adjusted_weight,
                      accuracy=weight_data['accuracy'],
                      avg_problem_score=weight_data['avg_problem_score'])
            
            # Debug information for Two Pointers to Sliding Window
            if source == "Two Pointers" and target == "Sliding Window":
                print("\n====== DEBUG: Two Pointers → Sliding Window Edge ======")
                print(f"Base weight: {weight}")
                print(f"Sliding Window accuracy: {weight_data['accuracy']}")
                print(f"Sliding Window avg_problem_score: {weight_data['avg_problem_score']}")
                print(f"Component1 (1-accuracy)*0.4: {weight_data['component1']}")
                print(f"Component2 (1-avg_score)*0.3: {weight_data['component2']}")
                print(f"Component3 (time factor)*0.3: {weight_data['component3']}")
                print(f"Dynamic weight: {dynamic_component}")
                print(f"Final adjusted weight: {dynamic_component} * {weight} = {adjusted_weight}")
                print("=========================================================")
    
    # Create graph layout (to clearly show the path)
    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    
    # Manually adjust positions to make the path clearer
    if 'Two Pointers' in pos and 'Sliding Window' in pos and 'Dynamic Programming 1D' in pos:
        # Make these points align in a more straight line
        pos['Two Pointers'][1] += 0.1
        pos['Sliding Window'][1] += 0.05
        pos['Dynamic Programming 1D'][1] -= 0.05
    
    # Create a large image for display
    plt.figure(figsize=(24, 18))
    
    # Node color mapping
    node_colors = {}
    for node in G.nodes():
        if node in strong_categories:
            node_colors[node] = 'green'
        elif node in weak_categories:
            node_colors[node] = 'red'
        elif dijkstra_result['path'] and node in dijkstra_result['path']:
            node_colors[node] = 'orange'
        else:
            node_colors[node] = 'lightblue'
    
    # Edge style mapping
    edge_colors = {}
    edge_widths = {}
    for u, v in G.edges():
        if dijkstra_result['path'] and u in dijkstra_result['path'] and v in dijkstra_result['path'] and dijkstra_result['path'].index(u) + 1 == dijkstra_result['path'].index(v):
            edge_colors[(u, v)] = 'purple'
            edge_widths[(u, v)] = 4.0
        else:
            edge_colors[(u, v)] = 'gray'
            edge_widths[(u, v)] = 1.0
    
    # Draw nodes
    for node in G.nodes():
        nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                              node_color=node_colors[node], 
                              node_size=2000, alpha=0.8)
    
    # Draw edges
    for u, v in G.edges():
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                             edge_color=edge_colors[(u, v)], 
                             width=edge_widths[(u, v)], alpha=0.7, arrowsize=20)
    
    # Draw node labels
    node_labels = {}
    for node in G.nodes():
        if node in metrics:
            accuracy = metrics[node]['accuracy'] * 100
            node_labels[node] = f"{node}\n({accuracy:.1f}%)"
        else:
            node_labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_weight='bold')
    
    # Draw edge weight labels with NetworkX's built-in function
    # But create separate labels for optimal and non-optimal paths
    optimal_edge_labels = {}
    other_edge_labels = {}
    
    for u, v, data in G.edges(data=True):
        base = data['base_weight']
        dynamic = data['dynamic_component']
        adjusted = data['adjusted_weight']
        
        # Check if this edge is part of the optimal path
        is_optimal_path = dijkstra_result['path'] and u in dijkstra_result['path'] and v in dijkstra_result['path'] and dijkstra_result['path'].index(u) + 1 == dijkstra_result['path'].index(v)
        
        if is_optimal_path:
            optimal_edge_labels[(u, v)] = f"Base: {base:.1f}\nDynamic: {dynamic:.2f}\nFinal: {adjusted:.2f}"
        else:
            other_edge_labels[(u, v)] = f"{base:.1f}→{adjusted:.2f}"
    
    # Draw labels in two separate calls to avoid overlap
    nx.draw_networkx_edge_labels(G, pos, edge_labels=other_edge_labels, font_size=8, 
                                font_color='darkgray', alpha=0.9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=optimal_edge_labels, font_size=10,
                                font_color='black', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
    
    # Special debug for Two Pointers to Sliding Window edge
    if dijkstra_result['path'] and "Two Pointers" in dijkstra_result['path'] and "Sliding Window" in dijkstra_result['path']:
        if dijkstra_result['path'].index("Two Pointers") + 1 == dijkstra_result['path'].index("Sliding Window"):
            u, v = "Two Pointers", "Sliding Window"
            data = G.get_edge_data(u, v)
            print(f"\nDEBUG: Two Pointers → Sliding Window edge data:")
            print(f"Base weight: {data['base_weight']}")
            print(f"Dynamic component: {data['dynamic_component']}")
            print(f"Adjusted weight: {data['adjusted_weight']}")
            print(f"Label used: {optimal_edge_labels.get((u, v), 'No label')}")
    
    # Add title and legend
    plt.title("Dijkstra Algorithm Learning Path Selection Process", fontsize=20)
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], color='purple', linewidth=4, label='Optimal Path'),
        Line2D([0], [0], color='gray', linewidth=1, label='Other Connections'),
        mpatches.Patch(color='green', label='Strong Categories (Start)'),
        mpatches.Patch(color='red', label='Weak Categories (Target)'),
        mpatches.Patch(color='orange', label='Path Intermediate Nodes'),
        mpatches.Patch(color='lightblue', label='Other Categories')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=14)
    
    # Add optimal path and distance information at the bottom
    if dijkstra_result['path']:
        path_str = " → ".join(dijkstra_result['path'])
        distance_str = f"{dijkstra_result['distance']:.3f}"
        
        plt.figtext(0.5, 0.02, f"Optimal Learning Path: {path_str}", 
                   ha="center", fontsize=16, 
                   bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8})
        
        plt.figtext(0.5, 0.06, f"Path Total Distance: {distance_str}", 
                   ha="center", fontsize=14, 
                   bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8})
    
    # Add weight calculation formula explanation
    formula_text = "Dynamic Weight Calculation Formula:\n"
    formula_text += "weight = (1-accuracy)*0.4 + (1-avg_score)*0.3 + time_factor*0.3\n"
    formula_text += "• accuracy: User's success rate in the category\n"
    formula_text += "• avg_score: User's average problem score in the category\n"
    formula_text += "• time_factor: Time factor (assumed 0 in this example)\n\n"
    formula_text += "Total weight = Dynamic Weight × Base Weight"
    
    plt.figtext(0.98, 0.02, formula_text, 
               va="bottom", ha="right", fontsize=12, 
               bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8})
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_dynamic_weights(metrics, solved_problems, problem_categories):
    """Visualize the dynamic weight components for each category"""
    categories = list(metrics.keys())
    
    # Calculate weight components for each category
    weight_data = []
    for category in categories:
        data = compute_weight(category, metrics, solved_problems, problem_categories)
        weight_data.append({
            'category': category,
            'accuracy': data['accuracy'],
            'avg_problem_score': data['avg_problem_score'],
            'component1': data['component1'],
            'component2': data['component2'],
            'component3': data['component3'],
            'weight': data['weight']
        })
    
    # Sort by total weight
    weight_data.sort(key=lambda x: x['weight'], reverse=True)
    
    # Extract data
    sorted_categories = [d['category'] for d in weight_data]
    accuracies = [d['accuracy'] for d in weight_data]
    avg_scores = [d['avg_problem_score'] for d in weight_data]
    components1 = [d['component1'] for d in weight_data]
    components2 = [d['component2'] for d in weight_data]
    components3 = [d['component3'] for d in weight_data]
    weights = [d['weight'] for d in weight_data]
    
    # Create charts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [1, 1.5]})
    
    # Generate x coordinates
    x = np.arange(len(sorted_categories))
    width = 0.35
    
    # Upper part: Accuracy and average scores
    ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
    ax1.bar(x + width/2, avg_scores, width, label='Average Problem Score', color='lightgreen')
    
    # Configure chart
    ax1.set_ylabel('Value')
    ax1.set_title('Category Accuracy and Average Problem Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Lower part: Weight components and total weight
    bar_width = 0.2
    ax2.bar(x - bar_width, components1, bar_width, label='(1-Accuracy)*0.4', color='salmon')
    ax2.bar(x, components2, bar_width, label='(1-Avg Score)*0.3', color='lightblue')
    ax2.bar(x + bar_width, components3, bar_width, label='Time Factor*0.3', color='lightgreen')
    ax2.bar(x + bar_width*2, weights, bar_width, label='Total Dynamic Weight', color='purple')
    
    # Configure chart
    ax2.set_ylabel('Weight Value')
    ax2.set_title('Dynamic Weight Component Breakdown by Category')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sorted_categories, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add weak and strong category markers
    for i, category in enumerate(sorted_categories):
        # Mark weak categories
        if metrics[category]['accuracy'] < 0.6:
            ax1.annotate('Weak Category', xy=(i, 1.05), xycoords=('data', 'axes fraction'),
                       ha='center', va='bottom', color='red', weight='bold')
        # Mark strong categories
        if metrics[category]['accuracy'] >= 0.8:
            ax1.annotate('Strong Category', xy=(i, 1.05), xycoords=('data', 'axes fraction'),
                       ha='center', va='bottom', color='green', weight='bold')
    
    plt.tight_layout()
    plt.show()
    

def main():
    # Get user category metrics and solved problems
    metrics, solved_problems, strong_categories, weak_categories, user_progress = calculate_category_metrics()
    problem_categories = get_problem_categories()
    
    print(f"Strong Categories: {strong_categories}")
    print(f"Weak Categories: {weak_categories}")
    
    # Get graph structure
    graph = load_graph_structure()
    
    # Simulate Dijkstra algorithm process (according to actual implementation)
    dijkstra_result = simulate_dijkstra(graph, metrics, solved_problems, problem_categories, 
                                       strong_categories, weak_categories)
    
    # Display optimal path
    if dijkstra_result['path']:
        print(f"\nOptimal Learning Path: {' -> '.join(dijkstra_result['path'])}")
        print(f"Path Distance: {dijkstra_result['distance']:.3f}")
    else:
        print("\nNo valid path found")
    
    # Print Dijkstra algorithm steps in console
    print("\n============== DIJKSTRA ALGORITHM EXECUTION STEPS ==============")
    for i, step in enumerate(dijkstra_result['steps'], 1):
        # Current node being processed
        node = step['current_node']
        
        if step.get('reached_target'):
            print(f"\nStep {i}: Process {node} (TARGET REACHED)")
            print(f"  Path Found: {' -> '.join(step.get('path', []))}")
            print(f"  Distance: {step['current_distance']:.3f}")
            continue
        
        # Check updated edges
        updated_edges = [e for e in step['edges_considered'] if e['updated']]
        
        print(f"\nStep {i}: Process {node}")
        if len(step['edges_considered']) == 0:
            print("  No neighbors to process")
            continue
            
        print("  Neighbors considered:")
        for edge in step['edges_considered']:
            neighbor = edge['to']
            base_weight = edge['base_weight']
            dynamic_component = edge['dynamic_component']
            adjusted_weight = edge['adjusted_weight']
            new_distance = edge['new_distance']
            accuracy = edge['accuracy']
            avg_score = edge['avg_problem_score']
            
            # Weight calculation components
            weight_calc1 = edge['weight_calculation']['component1']
            weight_calc2 = edge['weight_calculation']['component2']
            
            if edge['updated']:
                print(f"  → {neighbor} (UPDATED)")
            else:
                print(f"  → {neighbor} (not updated)")
                
            print(f"    Base Weight: {base_weight:.2f}")
            print(f"    Dynamic Weight Calculation:")
            print(f"      (1-{accuracy:.2f})*0.4 + (1-{avg_score:.2f})*0.3 = {weight_calc1:.3f} + {weight_calc2:.3f} = {dynamic_component:.3f}")
            print(f"    Final Weight: {base_weight:.2f} × {dynamic_component:.3f} = {adjusted_weight:.3f}")
            print(f"    New Distance: {step['current_distance']:.3f} + {adjusted_weight:.3f} = {new_distance:.3f}")
            
            if edge['updated']:
                print(f"    → Distance updated from ∞ to {new_distance:.3f}")
    
    print("\n============== WEIGHT COMPONENTS BY CATEGORY ==============")
    # Sort categories by accuracy
    sorted_by_accuracy = sorted([(cat, metrics[cat]['accuracy']) for cat in metrics], 
                              key=lambda x: x[1], reverse=True)
    
    print("\nCategories by Accuracy (High to Low):")
    for i, (category, accuracy) in enumerate(sorted_by_accuracy, 1):
        weight_data = compute_weight(category, metrics, solved_problems, problem_categories)
        accuracy_pct = accuracy * 100
        status = ""
        if category in strong_categories:
            status = " [STRONG]"
        elif category in weak_categories:
            status = " [WEAK]"
        
        print(f"{i}. {category}{status}: {accuracy_pct:.1f}% ({metrics[category]['correct']}/{metrics[category]['total']})")
        print(f"   Dynamic Weight: {weight_data['weight']:.3f}")
        print(f"   Components: (1-{accuracy:.2f})*0.4={weight_data['component1']:.3f}, " + 
              f"(1-{weight_data['avg_problem_score']:.2f})*0.3={weight_data['component2']:.3f}")
    
    # Display each category's weight
    print("\nSorted by Dynamic Weight (High to Low):")
    weight_data_list = []
    for category in metrics:
        weight_data = compute_weight(category, metrics, solved_problems, problem_categories)
        weight_data_list.append((category, weight_data))
    
    # Sort by weight (higher weights first)
    sorted_weight_data = sorted(weight_data_list, key=lambda x: x[1]['weight'], reverse=True)
    
    for i, (category, data) in enumerate(sorted_weight_data, 1):
        status = ""
        if category in strong_categories:
            status = " [STRONG]"
        elif category in weak_categories:
            status = " [WEAK]"
            
        print(f"{i}. {category}{status}: Weight={data['weight']:.3f}, Accuracy={data['accuracy']:.2f}, Score={data['avg_problem_score']:.2f}")
    
    # Visualize Dijkstra process (without steps text on the graph)
    visualize_dijkstra_process(graph, metrics, solved_problems, problem_categories, 
                             strong_categories, weak_categories, dijkstra_result)
    
    # Visualize dynamic weight analysis
    visualize_dynamic_weights(metrics, solved_problems, problem_categories)

if __name__ == "__main__":
    main() 