# Compliling on Linux  
## 1.Install PCL
* Run these commands to install dependencies first:
```
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
```           
* Then visit [PCL Docs](https://pcl.readthedocs.io/projects/tutorials/en/latest/compiling_pcl_posix.html) to build and install PCL.
## 2. Install igraph
* Tutorials of install igraph can be found at [igraph Reference Manual](https://igraph.org/c/doc/igraph-Installation.html).
## 3. Build MAC
- Option 1 (purely on the command line): Use CMake to generate Makefiles and then `make`.
    - You can simply run
      ```
      $ cd path-to-root-dir-of-MAC
      $ mkdir Release
      $ cd Release
      $ cmake -DCMAKE_BUILD_TYPE=Release ..
      $ make
      ```
- Option 2: Use any IDE that can directly handle CMakeLists files to open the `CMakeLists.txt` in the **root** directory of MAC. Then you should have obtained a usable project and just build it. I recommend using [CLion](https://www.jetbrains.com/clion/).
- NOTICE: Please compile in **RELEASE** mode!
## 4. DEMO
* Keep the complied file and ``demo`` in the same folder, then run the command to show an example:
```
./MAC --demo
```
* You should see terminal output like this:
```
Start registration.
 graph construction: 4.50404
 coefficient computation: 0.607463
0.417615->min(0.417615 0.798804 1.69403)
 clique computation: 0.457012
 clique selection: 0.0262053
 hypothesis generation & evaluation: 0.0609555
9762 : 528867
14 154.948
3.56584 3.4174 3.76196 3.62198 4.8172 5.63182 4.99794 4.46311 4.50627 5.37475 5.32872 5.37915 4.23573 5.59911 
  0.978491  -0.126114    0.16325   0.837719
   0.13669   0.989074 -0.0552203   0.161821
 -0.154502  0.0763472   0.985038   0.107174
         0          0          0          1
Press space to register.
```
* Two visualizations will be shown:
  - Selected matches:
    ![2023-11-14 19-47-40屏幕截图](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques/assets/42647472/6c702df5-285b-40c6-a212-660ab61fd454)

  - Registration:   
    The original status of input point clouds:
    ![2023-11-14 19-48-03屏幕截图](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques/assets/42647472/d989fd1e-6055-4db5-b60f-4e310547c145)
    The status after transformation (Do not forget to press the **SPACE** key):
    ![2023-11-14 19-48-21屏幕截图](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques/assets/42647472/b66f92fc-5eab-4df7-9f6e-40c480621faf)
