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

### Compliling on Windows  


### Compliling on Linux  
* Run these commands to install dependencies first:

           sudo apt-get update
           sudo apt-get install git build-essential linux-libc-dev -y
           sudo apt-get install cmake -y
           sudo apt-get install libusb-1.0-0-dev libusb-dev libudev-dev -y
           sudo apt-get install mpi-default-dev openmpi-bin openmpi-common -y
           sudo apt-get install libflann1.9 libflann-dev -y
           sudo apt-get install libeigen3-dev -y
           sudo apt-get install libboost-all-dev -y
           sudo apt-get install libvtk7.1p-qt libvtk7.1p libvtk7-qt-dev -y
           sudo apt-get install libqhull* libgtest-dev -y
           sudo apt-get install freeglut3-dev pkg-config -y
           sudo apt-get install libxmu-dev libxi-dev -y
           sudo apt-get install mono-complete -y
           sudo apt-get install openjdk-8-jdk openjdk-8-jre -y
           
* Then visit [PCL Docs](https://pcl.readthedocs.io/projects/tutorials/en/latest/compiling_pcl_posix.html) to build and install PCL.
* Tutorials of install igraph can be found at [igraph Reference Manual](https://igraph.org/c/doc/igraph-Installation.html).
## Results

## Citation
