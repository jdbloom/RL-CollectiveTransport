# RL-Collective Transport Running Via Vicon
Collective transport done with reinforcement learning, integrating simulate models into the Vicon tracking environment.

To run the simulation with Vicon, you must first install the [ARGoS](https://github.com/ilpincy/argos3), [KheperaIV](https://github.com/ACatastrophicBing/argos3-kheperaiv-gripper-module), and [Vicon](https://github.com/NESTLab/Vicon) libraries onto your computer.

# Running with Vicon
To begin, the project needs to be compiled with the option VICON=ON in the cmake, or by using 'network_build.sh', and your computer and the robots must be connected to the **NESTLab** wifi

The step by step process to run this environment are as follows :

1. Connect to **NESTLab** wifi
2. Power on Vicon tracking cameras, robots, and confirm robots and objects are being tracked on Vicon
3. Begin running your .argos file following the same loop function and physics engine format found in 'collectiveRLtransportReal0.argos', with 'simulate_robots' set to false
4. Begin running your python command following the same format as used for a simulated run
5. If you have visualizations on, wait for window to show and begin connecting khepera's to your computer by ssh'ing into each robot and running 'GAARARemoteDrive'. If visualizations are off, wait for your robots, obstacles, and gate to be simulated or recognized as being tracked following the terminal ARGoS logs
6. Press play in the ARGoS simulation window, if you are not using visualizations, run visualizations and start over.

# Running ARGoS
Run 'argos3 -c argos/collectiveRLTransportReal0.argos' for only four robots and and object being tracked using Vicon.

There are three additional flags you can modify to enable tracking using Vicon:
- 'simulate_robots' Default is true, to track the robots this must be false
- 'simulate_obstacles' Default is true, to track any obstacles you place in the field, set this to false 
- 'simulate_gate' Default is true, to track a gate in the field, set this to false; Gates cannot properly be tracked as of now, this setting is here for future implementation.

Visualization is required to run your robots in real, it is also recommended to track the robot entity names while running to properly understand what your simulation is doing.
