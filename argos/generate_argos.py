import re
import argparse
import os

def generate_argos(num_robots="1", max_num_robot_failures="1", chance_failure="0.5", num_episodes="1000",
                   pytorch_port="tcp://localhost:55555", argos_filename="collectiveRlTransport.argos",
                   alphabet_size='4'):
    containing_folder = os.path.dirname(os.path.realpath(__file__))
    template_file = os.path.join(containing_folder, "collectiveRlTransport_template.argos")
    with open(template_file, 'r') as f:
        filestring = f.read()
        filestring = re.sub(r'\$\$num_robots\$\$', num_robots, filestring)
        filestring = re.sub(r'\$\$max_robot_failures\$\$', max_num_robot_failures, filestring)
        filestring = re.sub(r'\$\$chance_failure\$\$', chance_failure, filestring)
        filestring = re.sub(r'\$\$num_episodes\$\$', num_episodes, filestring)
        filestring = re.sub(r'\$\$pytorch_port\$\$', pytorch_port, filestring)
        filestring = re.sub(r'\$\$alphabet_size\$\$', alphabet_size, filestring)

    argos_filename = os.path.join(containing_folder, argos_filename)
    with open(argos_filename, 'w') as f:
        f.write(filestring)

parser = argparse.ArgumentParser()
parser.add_argument("--num_robots", default="1")
parser.add_argument("--max_num_robot_failures", default="1")
parser.add_argument("--chance_failure", default="0.5")
parser.add_argument("--num_episodes", default="5000")
parser.add_argument("--pytorch_port", default="55555")
parser.add_argument("--argos_filename", default="collectiveRlTransport.argos")
parser.add_argument("--alphabet_size", default="4")
args = parser.parse_args()


generate_argos(**vars(args))
