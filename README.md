# 3D-Registration-with-Maximal-Cliques
Source code of CVPR 2023 paper  

## Introduction  

![](figures/pipeline.png)

## Repository layout  
The repository contains a `CMakeLists.txt` file (in the root directory of the repository) that serves as an anchor for  
configuring and building programs, and a set of subfolders:  
* [`code`](https://github.com/chsl/PLADE/tree/master/code) - source code of PLADE implementation.  
* [`sample_data`](https://github.com/chsl/PLADE/tree/master/sample_data) - two pairs of test point clouds.  
  
The core registration function is defined in [plade.h](./code/PLADE/plade.h).

## Build
MAC depends on [PCL](https://github.com/PointCloudLibrary/pcl/tags) (`>= 1.10.1`) and [igraph](https://github.com/igraph/igraph/tags)(`=0.9.9`). Please install these libraries first.

To build PLADE, you need [CMake](https://cmake.org/download/) (`>= 3.23`) and, of course, a compiler that supports `>= C++11`. The code in this repository has been tested on Windows (MSVC `=2022` `x64`), and Linux (GCC `=10.4.0`). Machines nowadays typically provide higher [support](https://en.cppreference.com/w/cpp/compiler_support), so you should be able to build MAC on almost all platforms.

### Windows version  
Please refer to [Compiling on Windows](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques/blob/main/Windows/readme.md) for details.

### Linux version
Please refer to [Compiling on Linux](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques/blob/main/Linux/readme.md) for details.

## Datasets

### LoInlierRatio dataset
Download links can be found in [LoInlierRatio](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques/blob/main/LoInlierRatio/readme.md).

## Results

## Citation
