cmake_minimum_required(VERSION 3.27)
# include(FetchContent)
project(lpcpp LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS Yes)

# FetchContent_Declare(
#     tracy
#     GIT_REPOSITORY https://github.com/wolfpld/tracy.git
#     GIT_TAG v0.13.1
#     GIT_SHALLOW TRUE
#     GIT_PROGRESS TRUE
# )
#
# FetchContent_MakeAvailable(tracy)

file(GLOB lpcpp_SRC "src/*.cpp" "include/*.hpp")
list(FILTER lpcpp_SRC EXCLUDE REGEX "gpu")

add_executable(lpcpp ${lpcpp_SRC})
target_include_directories(lpcpp PUBLIC "${CMAKE_SOURCE_DIR}/include")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(lpcpp PUBLIC ${OpenCV_LIBS} -ltbb)
# target_link_libraries(lpcpp PUBLIC TracyClient ${OpenCV_LIBS} -ltbb)
