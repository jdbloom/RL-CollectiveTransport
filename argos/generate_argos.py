import re
import argparse
import os

def generate_argos(num_obstacles = "0", num_robots="1", max_num_robot_failures="1",
                   chance_failure="0.25", num_episodes="1000",
                   pytorch_port="tcp://localhost:55555",
                   argos_filename="collectiveRlTransport.argos",
                   use_gate = "0", gate_curriculum = "0",
                   seed = "123", gate_minimum="4",
                   gate_success_threshold="0.8", gate_success_window="20",
                   use_prisms="0", random_objs="0", test_prism="0",
                   arena_x="28", arena_y="18",
                   wall_ns_y="5.1", wall_ew_x="10.1",
                   goal_x="4.5", goal_radius="2",
                   episode_time="4500",
                   cylinder_radius="0.5",
                   object_shape="cylinder", rod_length="3.0", rod_width="0.5",
                   obstacle_radius="0.5", obstacle_height="0.5"):

    containing_folder = os.path.dirname(os.path.realpath(__file__))
    template_file = os.path.join(containing_folder, "collectiveRlTransport_template.argos")
    with open(template_file, 'r') as f:
        filestring = f.read()
        filestring = re.sub(r'\$\$seed\$\$', seed, filestring)
        filestring = re.sub(r'\$\$num_robots\$\$', num_robots, filestring)
        filestring = re.sub(r'\$\$max_robot_failures\$\$', max_num_robot_failures, filestring)
        filestring = re.sub(r'\$\$chance_failure\$\$', chance_failure, filestring)
        filestring = re.sub(r'\$\$num_episodes\$\$', num_episodes, filestring)
        filestring = re.sub(r'\$\$pytorch_port\$\$', pytorch_port, filestring)
        filestring = re.sub(r'\$\$num_obstacles\$\$', num_obstacles, filestring)
        filestring = re.sub(r'\$\$use_gate\$\$', use_gate, filestring)
        filestring = re.sub(r'\$\$gate_curriculum\$\$', gate_curriculum, filestring)
        filestring = re.sub(r'\$\$gate_minimum\$\$', gate_minimum, filestring)
        filestring = re.sub(r'\$\$gate_success_threshold\$\$', gate_success_threshold, filestring)
        filestring = re.sub(r'\$\$gate_success_window\$\$', gate_success_window, filestring)
        filestring = re.sub(r'\$\$use_prisms\$\$', use_prisms, filestring)
        filestring = re.sub(r'\$\$random_objs\$\$', random_objs, filestring)
        filestring = re.sub(r'\$\$test_prism\$\$', test_prism, filestring)
        filestring = re.sub(r'\$\$arena_x\$\$', arena_x, filestring)
        filestring = re.sub(r'\$\$arena_y\$\$', arena_y, filestring)
        filestring = re.sub(r'\$\$wall_ns_y\$\$', wall_ns_y, filestring)
        filestring = re.sub(r'\$\$wall_ew_x\$\$', wall_ew_x, filestring)
        filestring = re.sub(r'\$\$goal_x\$\$', goal_x, filestring)
        filestring = re.sub(r'\$\$goal_radius\$\$', goal_radius, filestring)
        filestring = re.sub(r'\$\$episode_time\$\$', episode_time, filestring)
        filestring = re.sub(r'\$\$cylinder_radius\$\$', cylinder_radius, filestring)
        filestring = re.sub(r'\$\$object_shape\$\$', object_shape, filestring)
        filestring = re.sub(r'\$\$rod_length\$\$', rod_length, filestring)
        filestring = re.sub(r'\$\$rod_width\$\$', rod_width, filestring)
        filestring = re.sub(r'\$\$obstacle_radius\$\$', obstacle_radius, filestring)
        filestring = re.sub(r'\$\$obstacle_height\$\$', obstacle_height, filestring)


    argos_filename = os.path.join(containing_folder, argos_filename)
    with open(argos_filename, 'w') as f:
        f.write(filestring)

parser = argparse.ArgumentParser()
parser.add_argument("--num_obstacles", default="0")
parser.add_argument("--num_robots", default="1")
parser.add_argument("--max_num_robot_failures", default="1")
parser.add_argument("--chance_failure", default="0.25")
parser.add_argument("--num_episodes", default="1000")
parser.add_argument("--pytorch_port", default="55555")
parser.add_argument("--argos_filename", default="collectiveRlTransport.argos")
parser.add_argument("--use_gate", default="0")
parser.add_argument("--gate_curriculum", default=0)
parser.add_argument("--seed", default="123")
parser.add_argument("--gate_minimum", default=4)
parser.add_argument("--gate_success_threshold", type=str, default="0.8")
parser.add_argument("--gate_success_window", type=str, default="20")
parser.add_argument('--use_prisms', type=str, default='0')
parser.add_argument('--random_objs', type=str, default='0')
parser.add_argument('--test_prism', type=str, default='0')
parser.add_argument('--arena_x', type=str, default='28')
parser.add_argument('--arena_y', type=str, default='18')
parser.add_argument('--wall_ns_y', type=str, default='5.1')
parser.add_argument('--wall_ew_x', type=str, default='10.1')
parser.add_argument('--goal_x', type=str, default='4.5')
parser.add_argument('--goal_radius', type=str, default='2')
parser.add_argument('--episode_time', type=str, default='4500')
parser.add_argument('--cylinder_radius', type=str, default='0.5')
parser.add_argument('--object_shape', type=str, default='cylinder')
parser.add_argument('--rod_length', type=str, default='3.0')
parser.add_argument('--rod_width', type=str, default='0.5')
parser.add_argument('--obstacle_radius', type=str, default='0.5')
parser.add_argument('--obstacle_height', type=str, default='0.5')

args = parser.parse_args()
print(args)

generate_argos(**vars(args))
