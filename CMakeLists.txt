cmake_minimum_required(VERSION 3.27)
project(lpcpp LANGUAGES CXX CUDA)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS Yes)

file(GLOB lpcpp_SRC "src/*.cpp" "src/*.cu" "include/*.hpp")
add_executable(lpcpp ${lpcpp_SRC})
target_include_directories(lpcpp PUBLIC "${CMAKE_SOURCE_DIR}/include")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(lpcpp ${OpenCV_LIBS} -ltbb)
