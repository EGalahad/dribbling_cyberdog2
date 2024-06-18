cmake_minimum_required(VERSION 3.5)
project(robot_interface)

add_compile_options(-std=c++14)
set(CMAKE_CXX_FLAGS "-O3")

set(PYTHON_EXECUTABLE "/SDCARD/miniconda3/envs/drl/bin/python")
set(PYTHON_LIBRARY "/SDCARD/miniconda3/envs/drl/lib/libpython3.8.dylib")
set(PYTHON_INCLUDE_DIR "/SDCARD/miniconda3/envs/drl/include/python3.8")


FIND_PACKAGE(lcm REQUIRED)
FIND_PACKAGE(pybind11 REQUIRED)
# find_package(PythonLibs 3.8 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})

add_subdirectory(../cyberdog_motor_sdk ${CMAKE_BINARY_DIR}/cyberdog_motor_sdk_build)

# Add the cyberdog_motor_sdk submodule directory to include paths
include_directories(../cyberdog_motor_sdk/include)

# Generate robot_interface Python module
pybind11_add_module(robot_interface robot_interface.cpp)
target_link_libraries(robot_interface PRIVATE cyber_dog_motor_sdk ${pybind11_LIBRARIES})
set_target_properties(robot_interface PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                                SUFFIX "${PYTHON_MODULE_EXTENSION}")
