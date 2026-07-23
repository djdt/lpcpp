# lpcpp

lpcpp processes inline microscopy videos to detect particles using OpenCL.

## Installation
lpcpp requires CMake, a compiler supporting C++23, and the [IntelTBB](https://www.intel.com/content/www/us/en/docs/onetbb/get-started-guide/2022-1/overview.html) and [OpenCV 4.13](https://docs.opencv.org/4.x/df/d65/tutorial_table_of_content_introduction.html) libraries.

If building on Windows using Visual Studio then OpenCV is expected at ```C:\opencv``` and ```C:\opencv\build\x64\vcxx\bin``` should be added to envrionments path.

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
To run lpcpp pass a inline microscopy file and any options ```lpcpp [OPTIONS] file [SUBCOMMANDS]```.

The following options are available:
```
lpcpp [OPTIONS] file [SUBCOMMANDS]


POSITIONALS:
  file TEXT:FILE REQUIRED     path to the captured OIM video

OPTIONS:
  -h,     --help              
  -o,     --output TEXT:(PATH(non-existing)) OR (DIR) 
                              specify the output directory, defaults to 'processed'
          --selection-metric ENUM:value in {averageIntensity->0,centralIntensity->1,sharpness->2} OR {0,1,2} [1]  
                              method of selecting the particle frame for processing
          --detection-mode ENUM:value in {absolute->2,dark->1,light->0} OR {2,1,0} [1]  
                              method of thresholding differences from the background
          --background INT:POSITIVE [1000]  
                              number of background frames used to determine initial mean and
                              std
          --track INT:POSITIVE [50]  
                              number of frames to track particles after last detection
          --distance FLOAT:POSITIVE [5]  
                              minimum distance between particles
          --zscore FLOAT:POSITIVE [3]  
                              number of std above the background mean to threshold
          --unsharp FLOAT:NONNEGATIVE [1]  
                              alpha value of the unsharp mask
          --capillary [FLOAT,FLOAT,FLOAT]:NONNEGATIVE [[0,0,0]]  
                              capillary position and radius <x> <y> <radius>. If 0, try to read
                              from video
          --draw              show video and detections
          --export-images     export an image of each particle
          --export-hdf5       export VTK compatible HDF5 data sets for each particle
  -v,     --version           display version and exit
          --create-config     write default values to a new config file at 'file'
          --config            read options from a config file

SUBCOMMANDS:
filter
  options for filtering particles
  
  
OPTIONS:
          --area [FLOAT,FLOAT]:NONNEGATIVE [[5,10000]]  
                              allowed particle area
          --aspect [FLOAT,FLOAT]:FLOAT in [0 - 1] [[0.5,1]]  
                              allowed particle aspect ratio
          --circularity [FLOAT,FLOAT]:FLOAT in [0 - 1] [[0.5,1]]  
                              allowed particle circularity
          --convexity [FLOAT,FLOAT]:FLOAT in [0 - 1] [[0.5,1]]  
                              allowed particle convexity
          --intensity [FLOAT,FLOAT]:NONNEGATIVE [[1000,1e+06]]  
                              allowed particle intensity (darkness)
          --radius [FLOAT,FLOAT]:NONNEGATIVE [[1,11000]]  
                              allowed particle radius
          --sharpness [FLOAT,FLOAT]:NONNEGATIVE [[0,0]]  
                              allowed particle sharpness
```
Filter options can be passed as ```--filter.area 0 100``` etc.

## Python Explorer

A Python GUI is available to explore particle data exported from lpcpp or the BRAVE scripts.
Data is show as size histograms, scatter plots and heatmaps of the capillary.

### Installation

1. Enter the scripts folder
   ```
   cd scripts
   ```
2. Install using pip or uv.
   ```
   pip install -e ilmex
   ```
   ```
   uv pip install -e ilmex
   ```
3. Run the script.
   ```
   ilmex
   ```
   ```
   uv run ilmex
   ```
