cmake_minimum_required(VERSION 3.27)
project(lpcpp
    VERSION 0.3.0
    LANGUAGES CXX
)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS Yes)

add_compile_definitions(CMAKE_PROJECT_VERSION="${CMAKE_PROJECT_VERSION}")

file(GLOB lpcpp_SRC "src/*.cpp" "include/*.hpp")

add_executable(lpcpp ${lpcpp_SRC})
include_directories(lpcpp PUBLIC "${CMAKE_SOURCE_DIR}/include")

find_package(OpenCV 5.0 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

target_include_directories(lpcpp PRIVATE ${OpenCV_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
target_link_libraries(lpcpp PRIVATE ${OpenCV_LIBS} HDF5::HDF5)
