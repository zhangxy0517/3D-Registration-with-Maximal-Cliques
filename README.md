# 3D-Registration-with-Maximal-Cliques
Source code of CVPR 2023 paper  

## Introduction  

![](figures/pipeline.png)

## Repository layout  
The repository contains a set of subfolders:  
* [`Linux`](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques/tree/main/Linux) - source code for Linux platform.  
* [`Windows`](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques/tree/main/Windows) - source code for Windows platform.
* [`demo`](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques/tree/main/demo) - test point clouds.
* [`LoInlierRatio`](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques/tree/main/LoInlierRatio) - Download links for LoInlierRatio dataset.


## Build
MAC depends on [PCL](https://github.com/PointCloudLibrary/pcl/tags) (`>= 1.10.1`) and [igraph](https://github.com/igraph/igraph/tags)(`=0.9.9`). Please install these libraries first.

To build MAC, you need [CMake](https://cmake.org/download/) (`>= 3.23`) and, of course, a compiler that supports `>= C++11`. The code in this repository has been tested on Windows (MSVC `=2022` `x64`), and Linux (GCC `=10.4.0`). Machines nowadays typically provide higher [support](https://en.cppreference.com/w/cpp/compiler_support), so you should be able to build MAC on almost all platforms.

### Windows version  
Please refer to [Compiling on Windows](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques/blob/main/Windows/readme.md) for details.

### Linux version
Please refer to [Compiling on Linux](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques/blob/main/Linux/readme.md) for details.

## Usages
### Usage 1: register a 'source' point cloud to a 'target' point cloud.

### Usage 2: register a bunch of point cloud pairs in a batch mode.

## Datasets
### U3M

### 3DMatch & 3DLoMatch

### KITTI

### ETH

### LoInlierRatio dataset
Download links can be found in [LoInlierRatio](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques/blob/main/LoInlierRatio/readme.md).

## Results

## Citation
