import json
import networkx as nx
import matplotlib.pyplot as plt
from category_graph import CategoryGraph
from recommendation_system import RecommendationSystem

def show_user_learning_path():
    """
    Analyze and display the learning path graph for the current user
    """
    # Initialize recommendation system and category graph
    recommender = RecommendationSystem()
    category_graph = CategoryGraph()
    
    # Get user's strong and weak categories
    strong_categories = recommender.get_user_strong_categories()[:2]  # Get top 2 strong categories
    weak_categories = recommender.get_user_weak_categories()[:3]    # Get top 3 weak categories
    
    # Ensure weak categories are not empty
    if not weak_categories:
        print("Note: No obvious weak categories found, using all categories for path analysis")
        weak_categories = list(recommender.progress['user_progress']['categories'].keys())
        # Ensure strong categories are not in weak categories
        for category in strong_categories:
            if category in weak_categories:
                weak_categories.remove(category)
    
    # Print user information
    user_id = recommender.progress.get('user_id', 'Unknown User')
    print(f"\n=== Learning Path Analysis for User '{user_id}' ===")
    print(f"Strong Categories: {', '.join(strong_categories)}")
    print(f"Weak Categories: {', '.join(weak_categories)}")
    
    # Get Dijkstra path
    path_and_distance = category_graph.get_next_category_via_dijkstra(
        start_categories=strong_categories,
        target_categories=weak_categories,
        user_progress=recommender.progress['user_progress']
    )
    
    path, distance = path_and_distance
    if path:
        print(f"\nRecommended Learning Path: {' -> '.join(path)}")
        print(f"Path Distance: {distance:.2f}")
        
        # Explain multi-source, multi-target path selection
        print("\n=== Multi-Source, Multi-Target Path Finding ===")
        print(f"The system considers multiple starting points (your strengths) and multiple")
        print(f"target points (areas to improve) simultaneously.")
        
        # Display start and target points
        print(f"\nStarting Points (strengths):")
        for i, category in enumerate(strong_categories, 1):
            cat_stats = recommender.progress['user_progress']['categories'].get(category, {
                'total_attempted': 0, 'correct_solutions': 0
            })
            success_rate = (cat_stats['correct_solutions'] / cat_stats['total_attempted'] * 100) if cat_stats['total_attempted'] > 0 else 0
            print(f"  {i}. {category} - Success Rate: {success_rate:.1f}%")
            
            # Mark if this category is the one selected for the path
            if path and path[0] == category:
                print(f"     → SELECTED as optimal starting point")
        
        print(f"\nTarget Points (areas to improve):")
        for i, category in enumerate(weak_categories, 1):
            cat_stats = recommender.progress['user_progress']['categories'].get(category, {
                'total_attempted': 0, 'correct_solutions': 0
            })
            success_rate = (cat_stats['correct_solutions'] / cat_stats['total_attempted'] * 100) if cat_stats['total_attempted'] > 0 else 0
            print(f"  {i}. {category} - Success Rate: {success_rate:.1f}%")
            
            # Mark if this category is in the path
            if path and category in path:
                idx = path.index(category)
                print(f"     → Appears in position {idx+1} of the selected path")
        
        # Explain the algorithm approach
        print("\nPath Selection Logic:")
        print("• The system runs Dijkstra's algorithm from all strong categories")
        print("• It considers all weak categories as potential targets")
        print("• Among all possible paths, it selects the one with the lowest total weight")
        print("• The weight is adjusted based on your performance in each category")
        
        # Next recommended category
        if len(path) > 1:
            next_category = path[1]
            print(f"\nNext Recommended Category: {next_category}")
            
            # Explain recommendation reasons
            print("\n=== Recommendation Rationale ===")
            print(f"Starting from: {path[0]} (one of your strengths)")
            if path[-1] in weak_categories:
                print(f"Target goal: {path[-1]} (one of your areas to improve)")
            else:
                print(f"Moving toward: {', '.join(weak_categories)}")
            
            # Calculate and explain dynamic weights
            print("\n=== Dynamic Weight Calculation ===")
            # Compute the weight factor for the next category
            cat_stats = recommender.progress['user_progress']['categories'].get(next_category, {
                'total_attempted': 0,
                'correct_solutions': 0
            })
            
            attempted = cat_stats['total_attempted']
            correct = cat_stats['correct_solutions']
            time = cat_stats.get('average_time', 0)
            
            # Calculate accuracy, time factors
            accuracy = correct / attempted if attempted > 0 else 0
            normalized_time = time / 1000  # Assuming 1000s is a reasonable max
            
            # Calculate average problem score for this category
            avg_problem_score = 0.0
            problem_scores_count = 0
            
            for problem_name, problem_data in recommender.progress['user_progress'].get('problems', {}).items():
                if problem_data.get('solutions', []):
                    # Get the latest solution score
                    latest_solution = problem_data['solutions'][-1]
                    score = latest_solution.get('score', 0.0)
                    
                    # Get problem category
                    problem_category = category_graph.get_problem_category(problem_name)
                    
                    # If problem belongs to the next category, include in calculation
                    if problem_category == next_category:
                        avg_problem_score += score
                        problem_scores_count += 1
            
            if problem_scores_count > 0:
                avg_problem_score /= problem_scores_count
            else:
                avg_problem_score = 0.5  # Default for categories with no solutions
            
            # Display the weight calculation components
            print(f"Category: {next_category}")
            print(f"• Accuracy Factor: {accuracy:.2f} (from {correct}/{attempted} problems)")
            print(f"• Average Problem Score: {avg_problem_score:.2f} (from {problem_scores_count} solved problems)")
            print(f"• Time Factor: {normalized_time:.2f} (normalized from {time}ms)")
            
            # Calculate the final weight using the same formula as in category_graph.py
            weight_component_1 = (1 - accuracy) * 0.4
            weight_component_2 = (1 - avg_problem_score) * 0.3
            weight_component_3 = normalized_time * 0.3
            computed_weight = weight_component_1 + weight_component_2 + weight_component_3
            
            print(f"\nWeight Calculation:")
            print(f"• (1 - Accuracy) * 0.4 = {weight_component_1:.2f}")
            print(f"• (1 - Avg Score) * 0.3 = {weight_component_2:.2f}")
            print(f"• Time Factor * 0.3 = {weight_component_3:.2f}")
            print(f"• Total Computed Weight: {computed_weight:.2f}")
            
            # Get the base weight from the graph
            base_weight = category_graph.graph[path[0]][next_category]
            print(f"• Base Graph Weight: {base_weight:.2f} (pre-defined relationship strength)")
            print(f"• Final Adjusted Weight: {computed_weight * base_weight:.2f} (Computed × Base)")
            
            # Calculate path selection reasoning
            print(f"\nWhy This Path Was Chosen:")
            print(f"• The algorithm combines pre-defined category relationships with your performance")
            print(f"  metrics to find the optimal path from {path[0]} to your target categories.")
            print(f"• The dynamic weight calculation considers your accuracy, problem scores, and time")
            print(f"  spent in each category to personalize the learning journey.")
            print(f"• This particular path {path[0]} → {next_category} was selected because it has the")
            print(f"  lowest overall weight among all possible paths from your strong categories")
            print(f"  to your weak categories.")
            
            # Get user performance in recommended category
            next_category_stats = recommender.progress['user_progress']['categories'].get(next_category, {
                'total_attempted': 0,
                'correct_solutions': 0
            })
            
            total = next_category_stats['total_attempted']
            correct = next_category_stats['correct_solutions']
            
            if total > 0:
                success_rate = (correct / total * 100)
                print(f"• Your success rate in the {next_category} category is {success_rate:.1f}% ({correct}/{total}),")
                
                if success_rate < 60:
                    print(f"  which is an area that needs improvement.")
                else:
                    print(f"  this category will help you build a knowledge bridge from your strengths to weaker areas.")
            else:
                print(f"• You haven't attempted problems in the {next_category} category yet,")
                print(f"  which presents a good opportunity to expand your algorithmic knowledge.")
            
            # Explain overall path logic
            if len(path) > 2:
                print(f"\nComplete Path Explanation:")
                print(f"The recommended path {' -> '.join(path)} is the optimal learning sequence based on graph algorithm analysis.")
                print(f"It follows the principle of progressive learning, starting from knowledge you've already mastered and gradually transitioning to areas you need to strengthen.")
        else:
            next_category = path[0]
            print(f"\nRecommended Category: {next_category}")
            print("\n=== Recommendation Rationale ===")
            print(f"Direct recommendation of {next_category} category is due to the algorithm not finding a suitable transition path.")
            print(f"We suggest focusing on strengthening this category directly.")
        
        # Display performance data for each category
        print("\n=== User Performance by Category ===")
        categories_data = recommender.progress['user_progress']['categories']
        for category, data in categories_data.items():
            total = data['total_attempted']
            correct = data['correct_solutions']
            success_rate = (correct / total * 100) if total > 0 else 0
            print(f"{category}: Success Rate {success_rate:.1f}% ({correct}/{total})")
            
        # Create visualization
        visualize_path(category_graph, path, recommender.progress['user_progress'], strong_categories, weak_categories)
    else:
        print("Could not find a valid learning path")

def visualize_path(category_graph, path, user_progress, strong_categories, weak_categories):
    """
    Visualize the learning path with dynamic weights
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Function to compute dynamic weights (same as in category_graph.py)
    def compute_weight(category):
        data = user_progress['categories'].get(category, {
            'total_attempted': 0,
            'correct_solutions': 0
        })
        attempted = data['total_attempted']
        correct = data['correct_solutions']
        time = data.get('average_time', 0)
        
        accuracy = correct / attempted if attempted > 0 else 0
        normalized_time = time / 1000
        
        avg_problem_score = 0.0
        problem_scores_count = 0
        
        for problem_name, problem_data in user_progress.get('problems', {}).items():
            if problem_data.get('solutions', []):
                latest_solution = problem_data['solutions'][-1]
                score = latest_solution.get('score', 0.0)
                problem_category = category_graph.get_problem_category(problem_name)
                
                if problem_category == category:
                    avg_problem_score += score
                    problem_scores_count += 1
        
        if problem_scores_count > 0:
            avg_problem_score /= problem_scores_count
        else:
            avg_problem_score = 0.5
        
        weight = (1 - accuracy) * 0.4 + (1 - avg_problem_score) * 0.3 + normalized_time * 0.3
        return weight
    
    # Dictionary to store dynamic edge weights
    dynamic_weights = {}
    base_weights = {}
    
    # Add edges with both base and dynamic weights
    for source, targets in category_graph.graph.items():
        for target, base_weight in targets.items():
            G.add_edge(source, target, weight=base_weight)
            base_weights[(source, target)] = base_weight
            
            # Compute dynamic weight if there's user data
            if source in user_progress.get('categories', {}) and target in user_progress.get('categories', {}):
                dynamic_weight = compute_weight(target) * base_weight
                dynamic_weights[(source, target)] = dynamic_weight
    
    # Create layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Draw graph
    plt.figure(figsize=(16, 12))
    
    # Draw all nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500, alpha=0.6)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, alpha=0.4, arrowsize=15)
    
    # Highlight strong categories
    if strong_categories:
        nx.draw_networkx_nodes(G, pos, nodelist=strong_categories, 
                              node_color='green', node_size=1800, alpha=0.8)
    
    # Highlight weak categories
    if weak_categories:
        nx.draw_networkx_nodes(G, pos, nodelist=weak_categories, 
                              node_color='red', node_size=1800, alpha=0.7)
    
    # Highlight the path
    if path and len(path) > 1:
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='purple', 
                              width=3, alpha=1, arrowsize=20)
        
        # Highlight path nodes (that aren't already highlighted as strong/weak)
        path_only_nodes = [node for node in path if node not in strong_categories and node not in weak_categories]
        if path_only_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=path_only_nodes, 
                                  node_color='orange', node_size=2000, alpha=0.9)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Display weights on path
    if path and len(path) > 1:
        edge_labels = {}
        for i in range(len(path)-1):
            source = path[i]
            target = path[i+1]
            if (source, target) in dynamic_weights:
                # Show both base and dynamic weights
                base = base_weights[(source, target)]
                dynamic = dynamic_weights[(source, target)]
                edge_labels[(source, target)] = f"{base:.1f}→{dynamic:.1f}"
            else:
                # Just show base weight if dynamic not available
                if source in category_graph.graph and target in category_graph.graph[source]:
                    edge_labels[(source, target)] = f"{category_graph.graph[source][target]:.1f}"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    
    # Add legend
    plt.plot([], [], color='purple', linewidth=3, label='Recommended Path')
    plt.plot([], [], color='gray', linewidth=1, label='Possible Paths')
    plt.plot([], [], marker='o', markersize=15, markerfacecolor='green', 
            markeredgecolor='none', linestyle='', label='Strong Categories')
    plt.plot([], [], marker='o', markersize=15, markerfacecolor='red', 
            markeredgecolor='none', linestyle='', label='Weak Categories')
    plt.plot([], [], marker='o', markersize=15, markerfacecolor='orange', 
            markeredgecolor='none', linestyle='', label='Path Intermediate Nodes')
    plt.plot([], [], marker='o', markersize=15, markerfacecolor='lightblue', 
            markeredgecolor='none', linestyle='', label='Other Categories')
    plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='gray')
    
    # Chart title
    plt.title("Algorithm Learning Path Graph with Multiple Sources and Targets", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    
    # Save image
    filename = "user_learning_path.png"
    plt.savefig(filename)
    print(f"\nPath graph saved as: {filename}")
    
    # Image description
    print("\nImage Legend:")
    print("- Purple lines: System-recommended optimal learning path")
    print("- Green nodes: Your strong categories (multiple starting points)")
    print("- Red nodes: Your weak categories (multiple target points)")
    print("- Orange nodes: Intermediate categories on the recommended path")
    print("- Light blue nodes: Other algorithm categories")
    print("- Numbers on connections: Base→Dynamic weights (lower values indicate stronger connections)")
    print("  • Base weight: Pre-defined relationship strength")
    print("  • Dynamic weight: Adjusted based on your performance")
    
    # Show image
    plt.show()

if __name__ == "__main__":
    show_user_learning_path() 