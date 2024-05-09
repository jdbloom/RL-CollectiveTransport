# Running on Vicon : 
## Installation / Setup
For running the real robots with the GAARA Gripping modules, clone the following repositories and build / install accordingly on your computer:

https://github.com/NESTLab/Vicon/tree/kheperaiv_gaara_turret

https://github.com/ACatastrophicBing/argos3-kheperaiv-gripper-module

In RLCollectiveTransport, do the following : 
$ mkdir build_vicon
$ cd build_vicon
$ cmake -DVICON=ON ../
$ make
$ cd ..

Now in the lab, turn on the Vicon cameras, the app, and enable all robots you will be setting up within your environment as well as all cylinders you will be using in the environment as well.
You do not need to include obstacle cylinders as the khepera's will be using their own proxiimity sensors and not the simulator's.

The grippable cylinder must be named "Cylinder_1" in order to be recognized by your master loop function.

The robot ID's are defined as the robots_used="" in the collectiveRLTransportReal0.argos parameters. Default is 1, 3, 5, and 10.

Calibrate each individual robot's orientation on the vicon by pressing space to pause and selecting each object to modify their rotational +x, +y, +z.

## Running After everything is all set up and calibrated

### First off, turn on the robots.

Now ssh into each robot you will be using, and 
$ cd GAARA

You will be running the script GAARA_RemoteDrive, this may not be the exact name for it, but it definitely starts with GAARA.

### Run the loop function on your computer
Follow the same format as you would to run a trained model with the python code.

For argos, run collectiveRLTransportReal0.argos
$ argos3 -c collectiveRLTransportReal0.argos

BEFORE running, on the robot's terminals, run 
$ ./GAARA_RemoteDrive

Press play on the argos window, to stop the robots from running, you have to completely close out of the argos window, to open the grippers / reset the environment, run the last 3 steps again.
