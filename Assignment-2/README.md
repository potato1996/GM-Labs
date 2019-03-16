# Assignment 2: Implicit Surface Reconstruction

Author: Dayou Du (dayoudu@nyu.edu)

Assignment base codes credits to [Daniele Panozzo](http://cs.nyu.edu/~panozzo/) and [Zhongshi Jiang](https://cs.nyu.edu/~zhongshi/)

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

```
ex2_bin <model.off>
```

Keyboard Functions:

- 1: Display the point cloud
- 2: Build up the constrain points (Section 1)
- 3: MLS Interpolation (Section 2)
- 4: Show the reconstructed surface (Section 3)
- 5: MLS Interpolation with better normal constrains (Optional Task 2)

Side Effect: When type `4`, the program will also generate `out.off` which contains the reconstructed mesh

## UI parameters

- resolution (r): The reconstruction grid will have a dimension of r x r x r
- polyDegree (k): The degree of the polynominal basis function
- mesh/radiu (m/h): Control how large is the wendlandRatio(h). The real wendlandRatio will be computed as (diagonal_of_point cloud)/(m/h).
- radiu/epsilon (h/e): Control how large is the fixed epsilon chosen(e). The real epsilon will be computed as wendlandRatio/(h/e).

## Results and Report

The reconstructed mesh result for all the provided **point cloud** inputs are generated in the `output/` directory.

The report mainly contains the screenshot for each step and a comparision between **Screened Possion Surface Reconstruction** (Optional Task 3)

## Travis-CI
Every submission must build on Linux before it can be graded/considered
complete. To check this, you will use Travis-CI, a tool for automatically
rebuilding your code each time you push it to GitHub.
