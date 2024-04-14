import pickle
from dataclasses import dataclass


@dataclass
class exp_data:
    reward:list 
    epsilon:list 
    termination:list 
    loss:list 
    force_magnitude:list 
    force_angle:list 
    average_force_vector:list 
    cyl_x_pos:list 
    cyl_y_pos:list 
    cyl_angle:list 
    gate_stats:list 
    obstacle_stats:list 
    gsp_reward:list 
    gsp_heading:list
    run_time:list
    robots_x_pos:list
    robots_y_pos:list
    robot_angle:list
    robot_failures:list

class data_logger:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.reward:list = []
        self.epsilon:list = []
        self.termination:list = []
        self.loss:list = []
        self.force_magnitude:list = []
        self.force_angle:list = []
        self.average_force_vector:list = []
        self.cyl_x_pos:list = []
        self.cyl_y_pos:list = []
        self.cyl_angle:list = []
        self.gate_stats:list = []
        self.obstacle_stats:list = []
        self.gsp_reward:list = []
        self.gsp_heading:list = []
        self.run_time:list = []
        self.robots_x_pos:list = []
        self.robots_y_pos:list = []
        self.robot_angle:list = []
        self.robot_failures:list = []

    def writerow(
            self,
            rewards,
            epsilons,
            terminations,
            losses,
            force_magnitudes,
            force_angles,
            average_force_vectors,
            cyl_x_poses,
            cyl_y_poses,
            cyl_angles,
            gate_stats,
            obstacle_stats,
            gsp_rewards, gsp_headings,
            run_times,
            robots_x_poses,
            robots_y_poses,
            robot_angles,
            robot_failure
    ):
        self.reward.append(rewards)
        self.epsilon.append(epsilons)
        self.termination.append(terminations)
        self.loss.append(losses)
        self.force_magnitude.append(force_magnitudes)
        self.force_angle.append(force_angles)
        self.average_force_vector.append(average_force_vectors)
        self.cyl_x_pos.append(cyl_x_poses)
        self.cyl_y_pos.append(cyl_y_poses)
        self.cyl_angle.append(cyl_angles)
        self.gate_stats.append(gate_stats)
        self.obstacle_stats.append(obstacle_stats)
        self.gsp_reward.append(gsp_rewards)
        self.gsp_heading.append(gsp_headings)
        self.run_time.append(run_times)
        self.robots_x_pos.append(robots_x_poses)
        self.robots_y_pos.append(robots_y_poses)
        self.robot_angle.append(robot_angles)
        self.robot_failures.append(robot_failure)
    
    def write_to_file(self):
        data = exp_data(
            self.reward,
            self.epsilon,
            self.termination,
            self.loss,
            self.force_magnitude,
            self.force_angle,
            self.average_force_vector,
            self.cyl_x_pos,
            self.cyl_y_pos,
            self.cyl_angle,
            self.gate_stats,
            self.obstacle_stats,
            self.gsp_reward,
            self.gsp_heading,
            self.run_time,
            self.robots_x_pos,
            self.robots_y_pos,
            self.robot_angle,
            self.robot_failures,
        )

        with open(self.data_file_path, 'wb') as file:
            pickle.dump(data, file)
        
        print('SAVING DATA', self.data_file_path)



