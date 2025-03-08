import numpy as np
import matplotlib.pyplot as plt

# [Previous data dictionaries remain the same]
data_4_2 = {  # 4 Agents 2 Obstacles
    'IC': [90, 90.5, 91, 91.5, 93],
    'GK': [93, 94, 95, 96, 97],
    'GSP': [93, 94.5, 96, 97, 98],
    'GSP-N': [93, 94, 95.5, 96.5, 97],
    'AGSP-N': [94, 95, 96.5, 97.5, 99],
    'RGSP-N': [95, 96, 97, 98, 100]
}

data_4_4 = {  # 4 Agents 4 Obstacles
    'IC': [77, 80, 82, 83, 85],
    'GK': [78, 80, 82, 83, 85],
    'GSP': [82, 83, 85, 86, 87],
    'GSP-N': [82, 84, 86, 87, 88],
    'AGSP-N': [83, 85, 87, 89, 90],
    'RGSP-N': [82, 84, 86, 88, 90]
}

data_8_2 = {  # 8 Agents 2 Obstacles
    'IC': [91, 94, 95, 96, 97],
    'GK': [92, 94, 95, 96, 97],
    'GSP': [92, 93, 95, 96, 97],
    'GSP-N': [93, 94, 95, 96, 97],
    'AGSP-N': [94, 95, 97, 98, 99],
    'RGSP-N': [93, 94, 96, 97, 98]
}

data_8_4 = {  # 8 Agents 4 Obstacles
    'IC': [76, 78, 80, 85, 87],
    'GK': [75, 77, 80, 82, 83],
    'GSP': [77, 80, 85, 86, 88],
    'GSP-N': [78, 82, 85, 87, 88],
    'AGSP-N': [80, 82, 84, 86, 87],
    'RGSP-N': [79, 81, 83, 85, 87]
}

data_gate = {  # Gate Scenario
    'IC': [72, 74, 75, 77, 78],
    'GK': [70, 72, 73, 75, 77],
    'GSP': [73, 75, 77, 80, 83],
    'GSP-N': [75, 77, 80, 82, 85],
    'AGSP-N': [76, 79, 82, 84, 86],
    'RGSP-N': [75, 78, 81, 83, 85]
}

data_non_uniform_2 = {  # 2 Observations Scenario
    'IC': [30, 40, 84, 85, 85],
    'GK': [91, 92, 94, 95, 95],
    'GSP': [91, 92, 94, 96, 96],
    'GSP-N': [93, 96, 97, 98, 98],
    'RGSP-N': [90, 92, 95, 95, 95],
    'AGSP-N': [94, 95, 95, 97, 97]
}

data_non_uniform_4 = {  # 4 Observations Scenario
    'IC': [20, 40, 75, 81, 81],
    'GK': [80, 86, 87, 90, 90],
    'GSP': [71, 77, 86, 87, 87],
    'GSP-N': [80, 83, 93, 95, 95],
    'RGSP-N': [75, 88, 92, 92, 92],
    'AGSP-N': [85.5, 87, 88, 88, 88]
}

# Baseline values from the data table
baselines = {
    '4 Agents 2 Obstacles': 91,
    '4 Agents 4 Obstacles': 84,
    '8 Agents 2 Obstacles': 91,
    '8 Agents 4 Obstacles': 80,
    'Gate': 75,
    'Non-Uniform 2 Obstacles': 93,
    'Non-Uniform 4 Obstacles': 67
}

def create_box_plot_data(data_dict):
    boxes = []
    means = []
    algorithm_order = ['IC', 'GK', 'GSP', 'GSP-N', 'RGSP-N', 'AGSP-N']
    for key in algorithm_order:
        box_data = data_dict[key]
        boxes.append({
            'med': box_data[2],
            'q1': box_data[1],
            'q3': box_data[3],
            'whislo': box_data[0],
            'whishi': box_data[4]
        })
        means.append(np.mean(box_data))
    return boxes, means

def plot_boxes(scenarios, file, lower_bounds):
    for i, (data, title, pos) in enumerate(scenarios):
        # Create subplot with specific position and spacing
        ax = plt.subplot(1, len(scenarios), pos)
        
        # Create box plot
        boxes, means = create_box_plot_data(data)
        bp = ax.bxp(boxes, showfliers=False, patch_artist=True)
        
        # Add trend line
        x_positions = np.arange(1, len(means) + 1)
        ax.plot(x_positions, means, 'r-', linewidth=2, label='Mean Trend' if i == 0 else "")
        ax.scatter(x_positions, means, color='red', zorder=3)
        
        # Add baseline
        baseline = baselines[title]
        ax.axhline(y=baseline, color='black', linestyle='--', alpha=0.8, 
                label='Baseline' if i == 0 else "")
        
        # Customize box plots
        for box in bp['boxes']:
            box.set_facecolor('#2196f3')
            box.set_alpha(0.7)
        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='black')
        
        # Customize subplot
        ax.set_title(title)
        ax.set_ylim(lower_bounds, 100)
        ax.set_xticklabels(['IC', 'GK', 'GSP', 'GSP-N', 'RGSP-N', 'AGSP-N'], rotation=45)
        # ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only add y-label and legend to first subplot
        if i == 0:
            ax.set_ylabel('Average Success (%)')
            ax.legend()

    # Adjust the layout with specific margins
    plt.subplots_adjust(bottom=0.2, wspace=0.3)

    # Show plot
    # plt.show()

    # Optional: Save the plot
    plt.savefig(file, dpi=300, bbox_inches='tight')


scenarios = [
    (data_4_2, '4 Agents 2 Obstacles', 1),
    (data_4_4, '4 Agents 4 Obstacles', 2),
    (data_8_2, '8 Agents 2 Obstacles', 3),
    (data_8_4, '8 Agents 4 Obstacles', 4),
    (data_gate, 'Gate', 5),
    
]
non_uniform_scenarios = [
    (data_non_uniform_2, 'Non-Uniform 2 Obstacles', 1),
    (data_non_uniform_4, 'Non-Uniform 4 Obstacles', 2)
]
plt.figure(figsize=(20, 6))
plot_boxes(scenarios, 'box_plots_with_trendline.png', 69)
plt.clf()
plt.figure(figsize=(8, 6))
plot_boxes(non_uniform_scenarios, 'non-uniform_box_plots_with_trendline.png', 19)