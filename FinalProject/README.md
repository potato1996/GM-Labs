# Geometric Modeling: Final Project

Implement the Deformation algorithm from "Harmonic Coordinates for Character Articulation (Joshi et al.)". 2D version

## Prerequisites

- cmake
- libigl (which will be downloaded automatically by cmake)

## Build Guide

Simply build the code using cmake
```
mkdir build && cd build && cmake ..
make
```

## Usage

1. To run the program, simply type
```
final_bin
```

2. How to draw/read the object

There are two modes provided: the `FREE DRAW` mode and the `LOAD IMAGE` mode.

In the `FREE DRAW` mode, user can freely draw the desired object boundary after clicking the `RESET OBJECT` button. The object boundary will be in **YELLOW**. The ending point will be automatically attached to the starting point when they get close on the screen, so that the user can easily form a loop.

In the `LOAD IMAGE` mode, user will be able to load a image, as well as specify the object boundary within the image (if the object boundary is not specified or missing, will use the rectrangle boundary of the image by default).

3. How to draw the boundary

After clicking the `REDRAW CAGE` button, user can freely draw the desired cage boundary, which will be shown in **RED(()). The ending point will be automatically attached to the starting point when they get close on the screen, so that the user can easily form a loop.

Note that **we support selecting part of the object**. In this case the parts that outside the cage will not be modified.

4. How to deform the object

After finish drawing the cage boundary, the user can start deforming the object by clicking the `START DEFROM` button. When dragging the cage point around the user can see the object is being deformed in realtime.

Note that **we support re-draw the cage on a deformed object**. To do this one just simply click `REDRAW CAGE` after one round of deformation, and click `START DEFORM` again after drawing the new cage boundary.


## Results and Report

The screen shots are in the `screenshots` directory.