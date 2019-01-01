cmake_minimum_required (VERSION 2.6)
project (BenchmarkPlanner)

set(BenchmarkPlanner_VERSION_MAJOR 1)
set(BenchmarkPlanner_VERSION_MINOR 0)

set(CMAKE_BUILD_TYPE Release)

find_package(ompl REQUIRED)
find_library(OMPL_LIBRARIES ompl)
find_package(Eigen3 REQUIRED)
find_package(HDF5 COMPONENTS C CXX HL REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${HDF5_INCLUDE_DIRS}
  ${OMPL_INCLUDE_DIRS}
)

# add the executable
add_executable(benchmark
  src/benchmark_main.cpp
  src/HDF5Interface.cpp
  src/HeightMapClass.cpp
  src/MyGoalSampleableRegion.cpp
  src/MyOptimizationObjective.cpp
  src/MyRRTstar.cpp
  src/MySampler.cpp
  src/MySE3StateSpace.cpp
  src/MyStateValidityCheckerClass.cpp
  src/WindGrid.cpp
)

target_link_libraries(benchmark
  ${OMPL_LIBRARIES}
  ${Eigen_LIBRARIES}
  ${HDF5_CXX_LIBRARIES}
)
