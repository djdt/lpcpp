cmake_minimum_required(VERSION 3.27)
project(lpcpp
    VERSION 0.2.0
    LANGUAGES CXX
)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS Yes)


file(GLOB lpcpp_SRC "src/*.cpp" "include/*.hpp")

add_executable(lpcpp ${lpcpp_SRC})
target_include_directories(lpcpp PUBLIC "${CMAKE_SOURCE_DIR}/include")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(lpcpp PUBLIC ${OpenCV_LIBS} -ltbb)
