cmake_minimum_required(VERSION 3.0)
project(pcreg)

set(CMAKE_CXX_STANDARD 14)

find_package(PCL 1.9 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

aux_source_directory(src DIR_SRC)

add_executable(${PROJECT_NAME} ${DIR_SRC})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})
