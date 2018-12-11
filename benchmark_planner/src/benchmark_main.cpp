
#include <array>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <ompl/base/PlannerStatus.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/State.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/geometric/SimpleSetup.h>

#include "params.hpp"
#include "HDF5Interface.hpp"
#include "HeightMapClass.hpp"
#include "MyOptimizationObjective.hpp"
#include "MyProjection.hpp"
#include "MyRRTstar.hpp"
#include "MySE3StateSpace.hpp"
#include "MyStateValidityCheckerClass.hpp"
#include "WindGrid.hpp"

namespace intel_wind {

namespace planning {

namespace ob = ompl::base;
namespace og = ompl::geometric;

struct PlanningResult {
  bool solved = false;
  double planned_time = 0.0;
  double executed_time = 0.0;
};

PlanningResult plan_with_simple_setup(HDF5Interface::Sample input) {
  // set the wind grid
  WindGridGeometry geo;
  geo.min_x = 0.0;
  geo.min_y = 0.0;
  geo.min_z = 0.0;
  geo.nx = input.nx;
  geo.ny = input.ny;
  geo.nz = input.nz;
  geo.resolution_hor = input.resolution_horizontal;
  geo.resolution_ver = input.resolution_vertical;

  WindGrid wind_grid;
  wind_grid.setWindGrid(input.wind_x, input.wind_y, input.wind_z, geo);

  // set terrain
  HeightMapClass map = HeightMapClass();
  map.setHeightMapData(input.terrain, input.nx, input.nz * input.resolution_vertical, input.resolution_horizontal);

  // construct the state space we are planning in
  ob::StateSpacePtr space(new MySE3StateSpace());

  // set the state space bounds
  ob::RealVectorBounds bounds(3);
  bounds.setLow(0, map.getMinX());
  bounds.setHigh(0, map.getMaxX());
  bounds.setLow(1, map.getMinY());
  bounds.setHigh(1, map.getMaxY());
  bounds.setLow(2, 0);
  bounds.setHigh(2, map.getMaxZ());
  space->as<MySE3StateSpace>()->setBounds(bounds);

  // set the default projection, not sure if it actually is required
  space->registerDefaultProjection(
      ob::ProjectionEvaluatorPtr(
          new MyProjection(space, proj_cellsize)));

  // using simple setup
  og::SimpleSetup ss(space);

  // set the collision checking
  ss.setStateValidityChecker(ob::StateValidityCheckerPtr(
          new MyStateValidityCheckerClass(ss.getSpaceInformation(), map)));

  const double res = *std::min_element(std::begin(airplane_bb_param), std::end(airplane_bb_param) - 1)
      / (ss.getSpaceInformation()->getMaximumExtent()) * 0.5;
  ss.getSpaceInformation()->setStateValidityCheckingResolution(res);

  // Set optimization objectives
  ss.setOptimizationObjective(ob::OptimizationObjectivePtr(new MyOptimizationObjective(ss.getSpaceInformation(), wind_grid)));

  // set the planner
  MyRRTstar *my_rrtstar = new MyRRTstar(ss.getSpaceInformation());
  my_rrtstar->setRange(ss.getSpaceInformation()->getMaximumExtent() / 7.0);
  my_rrtstar->setGoalBias(0.05);
  my_rrtstar->setFocusSearch(false);
  // need to enable admissible cost to come, false sometimes leads to SEGFAULTs due to pruned goal states.
  my_rrtstar->setAdmissibleCostToCome(true);
  my_rrtstar->setCustomNearestNeighbor(false);
  my_rrtstar->setComputekNearestNeighbors2way(false);
  ss.setPlanner(ob::PlannerPtr(my_rrtstar));

  // define start and goal
  ob::ScopedState<> start(space);
  start.random();

  ob::ScopedState<> goal(space);
  goal.random();

  std::cout << "start" << std::endl;
  start.print(std::cout);
  std::cout << "goal" << std::endl;
  goal.print(std::cout);

  ss.setStartAndGoalStates(start, goal);

  // solve the path planning problem
  ob::PlannerStatus solved = ss.solve(10.0);

  if (solved) {
      std::cout << "Found solution:" << std::endl;
      // print the path to screen
      ss.getSolutionPath().print(std::cout);
  }
}

int benchmark(int argc, char *argv[]) {
  // open the specified file and the specified dataset in the file.
  HDF5Interface database;
  database.init("prediction.hdf5");

  // open the file with the start and goal configurations
  std::ifstream sg_file("start_goal_configurations.txt");

  // read the configurations from file
  std::vector<std::array<double, 8>> sg_configurations = {};
  std::array<double, 8> tmp = {};
  while (sg_file >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3] >> tmp[4] >> tmp[5] >> tmp[6] >> tmp[7]) {
    sg_configurations.push_back(tmp);
  }
  sg_file.close();

  for(auto const& tmp: sg_configurations) {
    printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7]);
  }


//  plan_with_simple_setup(database.getSample(0));
}

} // namespace planning

} // namespace intel_wind

int main (int argc, char *argv[]) {
  return intel_wind::planning::benchmark(argc, argv);
}
