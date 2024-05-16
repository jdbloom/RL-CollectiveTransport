import re
import argparse
import os

def generate_argos(num_obstacles = "0", num_robots="1", max_num_robot_failures="1",
                   chance_failure="0.25", num_episodes="1000",
                   pytorch_port="tcp://localhost:55555",
                   argos_filename="collectiveRlTransport.argos",
                   use_gate = "0", gate_curriculum = "0",
                   seed = "123", gate_minimum="4"):

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
        # filestring = re.sub(r'\$\$gate_curriculum_update_freq\$\$', gate_curriculum_update_freq, filestring)
        # filestring = re.sub(r'\$\$gate_curriculum_step\$\$', gate_curriculum_step, filestring)
        

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
# parser.add_argument("--gate_curriculum_update_freq", default=50)
# parser.add_argument("--gate_curriculum_step", default=0.25)

args = parser.parse_args()
print(args)

generate_argos(**vars(args))
