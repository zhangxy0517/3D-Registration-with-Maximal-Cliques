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

### Python implementation
We provide a simple demo in python, please refer to [Python_implement](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques/blob/main/Python_implement/README.md) for details.

## Usage:
* `--help` list all usages.
* `--demo` run the demo.
### Required args:
* `--output_path` output path for saving results. 
* `--input_path` input data path. 
* `--dataset_name`[3dmatch/3dlomatch/KITTI/ETH/U3M] dataset name.
* `--descriptor`[fpfh/fcgf/spinnet/predator] descriptor name. 
* `--start_index`(begin from 0) run from given index. 
### Optional args:
*[@-@]: `--lowInlierRatio` run test on the LoInlierRatio dataset.
*[^-^]: `--add_overlap` add the overlap input.
* `--no_logs` forbid generation of log files.

## Datasets
### U3M

### 3DMatch & 3DLoMatch

### KITTI

### ETH


## Results

## Citation
If you find this code useful for your work or use it in your project, please consider citing:

```shell
@InProceedings{zhang2023mac,
    author    = {Xiyu Zhang, Jiaqi Yang, Shikun Zhang and Yanning Zhang},
    title     = {3D Registration with Maximal Cliques},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {}
}
```
