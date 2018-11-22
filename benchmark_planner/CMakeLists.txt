cmake_minimum_required (VERSION 2.6)
project (BenchmarkPlanner)

set(BenchmarkPlanner_VERSION_MAJOR 1)
set(BenchmarkPlanner_VERSION_MINOR 0)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(ompl REQUIRED)
find_library(OMPL_LIBRARIES ompl)
find_package(Eigen3 REQUIRED)

include_directories(
  SYSTEM ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# add the executable
add_executable(benchmark
  src/benchmark_main.cpp
  src/HeightMapClass.cpp
  src/MyGoalSampleableRegion.cpp
  src/MyOptimizationObjective.cpp
  src/MyRRTstar.cpp
  src/MySampler.cpp
  src/MySE3StateSpace.cpp
  src/MyStateValidityCheckerClass.cpp
)

target_link_libraries(benchmark
  ${OMPL_LIBRARIES}
  ${Eigen_LIBRARIES}
)
