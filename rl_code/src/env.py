import math
import numpy as np

def calculate_gsp_reward(GSP, old_cyl_ang, cyl_ang, next_heading_gsp, num_robots):
    gsp_reward = []
    label = 0
    if GSP:
        # shift to get between 0 and 2 Pi
        old_cyl_ang += math.pi
        new_cyl_ang = cyl_ang + math.pi
        # check edge wrap at 0
        if old_cyl_ang < math.pi and new_cyl_ang > math.pi:
            diff = old_cyl_ang - new_cyl_ang + 2*math.pi
        # check edge wrap at 2 pi
        elif old_cyl_ang < math.pi and new_cyl_ang < math.pi:
            diff = new_cyl_ang - old_cyl_ang + 2*math.pi
        # otherwise we are not wrapping
        else:
            diff = old_cyl_ang - new_cyl_ang
        label=diff
        x1 = math.cos(diff)
        y1 = math.sin(diff)                        
        for i in range(num_robots):
                x2 = math.cos(next_heading_gsp[i] * math.pi)
                y2 = math.sin(next_heading_gsp[i] * math.pi)

                error = np.dot([x1, y1], [x2, y2])
                gsp_reward.append(-1 + error)
                

    else:
        gsp_reward = [0 for i in range(num_robots)]
    
    return gsp_reward, label