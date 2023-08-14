import matplotlib.pyplot as plt
import numpy as np

TD3_2_obstacles = [91, 95, 97]
TD3_4_obstacles = [82, 86, 85]
TD3_gate = [76, 78, 84]




#DDPG_2_obstacles = [91, 87, 94]
#DDPG_4_obstacles = [86, 73, 87]
#DDPG_gate = [77, 0, 80]

plt.plot(TD3_2_obstacles, label = '2 Obstacles')
plt.plot(TD3_4_obstacles, label = '4 Obstacles')
plt.plot(TD3_gate, label ='Gate')
x_tics = [0, 1, 2]
x_labels = ['IC', 'GK', 'GSP']
plt.xticks(x_tics, x_labels)
plt.legend()
plt.savefig('Communication_Comparison_plot.png')
#plt.show()

plt.clf()

ind = np.arange(3)
bar2 = plt.bar(ind-0.25, TD3_2_obstacles, 0.25, fill=False, hatch='///', label = '2 Obstacles')
bar1 = plt.bar(ind, TD3_4_obstacles, 0.25, fill=False, hatch='...', label = '4 Obstacles')
bar3 = plt.bar(ind+0.25, TD3_gate, 0.25, fill = False, hatch = 'xxx', label = 'Gate')
plt.xticks(x_tics, x_labels)
plt.ylim(70, 100)
plt.ylabel('Success Rate (%)', fontsize=30)
plt.legend()
plt.savefig('Communication_Comparison_bar.png')