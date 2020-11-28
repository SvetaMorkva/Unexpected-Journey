cmake_minimum_required(VERSION 3.17)
project(kpi_lab2)

find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 17)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45.0 COMPONENTS filesystem)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    add_executable(kpi_lab2 main.cpp ObjectDetector.cpp ObjectDetector.h fileutils.cpp fileutils.h)
    target_link_libraries(kpi_lab2 ${Boost_LIBRARIES} ${OpenCV_LIBS})
endif()