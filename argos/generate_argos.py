import re
import argparse

def generate_argos(num_robots="1", max_num_robot_failures="1", chance_failure="0.5", num_episodes="1000"):
    with open("collectiveRlTransport_template.argos", 'r') as f:
        filestring = f.read()
        filestring = re.sub(r'\$\$num_robots\$\$', num_robots, filestring)
        filestring = re.sub(r'\$\$max_robot_failures\$\$', max_num_robot_failures, filestring)
        filestring = re.sub(r'\$\$chance_failure\$\$', chance_failure, filestring)
        filestring = re.sub(r'\$\$num_episodes\$\$', num_episodes, filestring)

    with open("collectiveRlTransport.argos", 'w') as f:
        f.write(filestring)

parser = argparse.ArgumentParser()
parser.add_argument("--num_robots", default="1")
parser.add_argument("--max_num_robot_failures", default="1")
parser.add_argument("--chance_failure", default="0.5")
parser.add_argument("--num_episodes", default="1000")
args = parser.parse_args()

generate_argos(**vars(args))
