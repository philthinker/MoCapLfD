# MoCapLfD

The Motion Capture based Learning from Demonstration Method

Haopeng Hu

2023.05.31

![robot](https://github.com/philthinker/SAMP/blob/main/panda.jpg)

## Overview

### Greengrape

The MATLAB codes for learning the policies from the demonstration data.

- The codes are tested on MATLAB R2020b.
- Run "UNCORK.m" once you set "Greengrape" as your MATLAB workspace.

### ActionRecognition

The Python codes and the data set for the action recognition network.

### Data

Demo data.

- "ObjectA_MoCap.mat": The demo data for the MATLAB demo of Object A.
- "ObjectB_MoCap.csv": The demo data for the MATLAB demo of Object B.

The action recognition results are already implemented.

### Grape

The C++ codes for controlling the robot.

- The "libfranka 0.8.0" library is needed.
- There is an demo file "Toys_Grape6.cpp".
- Build it with CMake.

## Demos

- main_A_MoCap.m: Learning the policies of Object A.
- main_B_MoCap.m: Learning the policies of Object B.

Runing these codes may take seconds.



