# lpcpp

lpcpp processes inline microscopy videos to detect particles using OpenCL.

## Installation
lpcpp requires CMake, a compiler supporting C++23 and the following libraries: IntelTBB, OpenCV 4.

On windows OpenCV is expected at ```C:\opencv``` and ```C:\opencv\build\x64\vcxx\bin``` should be added to envrionments path.

1. Clone the repository
   ```
   git clone https://github.com/djdt/lpcpp
   cd lpcpp
   ```
3. Create a build directory
   ```
   mkdir build
   cd build
   ```
4. Run CMake to generate make files.
   ```
   cmake ..
   ```
5. Build the executable.
   ```
   cmake --build .
   ```

## Usage
To run lpcpp pass a inline microscopy file and any options ```lpcpp <FILE> [options]```.

The following options are available:
```
--background-frames, number of background frames used to determine initial mean and std, default = 1000
--particle-frames, number of frames to track particles after last detection, default = 50
--particle-distance, minimum distance between particles, default = 3
--zscore, number of std above the background mean to threshold, default = 3
--unsharp, apply an unsharp mask at this alpha, default = 1
--draw, show video and detections, default = false
--output, output directory for processed data
--export-images, export images of particles, default = false
--config, path to filter config, with lines: '<key> <min> <max>'
        valid keys are: 'area', 'aspect', circularity', 'convexity', 'intensity', 'radius', 'sharpness'.
        If no file exists, a default config file is created.
```
