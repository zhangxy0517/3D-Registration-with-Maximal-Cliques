# Compiling on Windows
Before comiling, please make sure that the environment variables of `PCL` and `igraph` have been correctly added to the system.

- Option 1 (purely on the command line): Use `nmake`(on Windows with Microsoft Visual Studio).
  - On Windows with Microsoft Visual Studio, use the `x64 Native Tools Command Prompt for VS XXXX` (**don't** use the x86 one), then
      ```
      $ cd path-to-root-dir-of-MAC
      $ mkdir Release
      $ cd Release
      $ cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ..
      $ nmake
      ```
- Option 2: Use any IDE that can directly handle CMakeLists files to open the `CMakeLists.txt` in the **root** directory of MAC. Then you should have obtained a usable project and just build it. I recommend using [CLion](https://www.jetbrains.com/clion/). For Windows users: your IDE must be set for `x64`.
- NOTICE: Please compile in **RELEASE** mode!
