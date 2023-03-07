# Compliling on Linux  
## 1.Install PCL
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
## 2. Install igraph
* Tutorials of install igraph can be found at [igraph Reference Manual](https://igraph.org/c/doc/igraph-Installation.html).
## 3. Build MAC
- Option 1 (purely on the command line): Use CMake to generate Makefiles and then `make`.
    - You can simply run
      ```
      $ cd path-to-root-dir-of-PLADE
      $ mkdir Release
      $ cd Release
      $ cmake -DCMAKE_BUILD_TYPE=Release ..
      $ make
      ```
- Option 2: Use any IDE that can directly handle CMakeLists files to open the `CMakeLists.txt` in the **root** directory of MAC. Then you should have obtained a usable project and just build it. I recommend using [CLion](https://www.jetbrains.com/clion/). 
