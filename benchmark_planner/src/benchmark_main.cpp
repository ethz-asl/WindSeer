
#include <iostream>
#include <vector>
#include <random>

#include <ompl/base/PlannerStatus.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/State.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/geometric/SimpleSetup.h>

#include "params.hpp"
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

int plan_with_simple_setup() {
  // domain settings
  const int nx = 64;
  const int ny = 64;
  const int nz = 64;
  const double resolution = 17.1875;
  const double max_z = 1100.0 / 96.0 * 64.0;
  std::mt19937 generator(std::random_device{}());

  // set the wind grid
  WindGridGeometry geo;
  geo.min_x = 0.0;
  geo.min_y = 0.0;
  geo.min_z = 0.0;
  geo.nx = nx;
  geo.ny = ny;
  geo.nz = nz;
  geo.resolution_hor = resolution;
  geo.resolution_ver = max_z / nz;

  std::uniform_real_distribution<float> distribution_wind(0.0, 1.0);
  std::vector<float> u, v, w;
  u.resize(nx*ny*nz, 0.0f);
  v.resize(nx*ny*nz, 0.0f);
  w.resize(nx*ny*nz, 0.0f);
  for (std::vector<float>::iterator it=u.begin(); it != u.end() ;++it) {
      *it = distribution_wind(generator);
  }
  for (std::vector<float>::iterator it=v.begin(); it != v.end() ;++it) {
      *it = distribution_wind(generator);
  }
  for (std::vector<float>::iterator it=w.begin(); it != w.end() ;++it) {
      *it = distribution_wind(generator);
  }

  WindGrid wind_grid;
  wind_grid.setWindGrid(u, v, w, geo);

  // generate a random terrain
  std::uniform_real_distribution<float> distribution(0.0, 0.8 * max_z);

  std::vector<float> terrain;
  terrain.resize(nx*ny, 0.0f);
  for (std::vector<float>::iterator it=terrain.begin(); it != terrain.end() ;++it) {
      *it = distribution(generator);
  }
  HeightMapClass map = HeightMapClass();
  map.setHeightMapData(terrain, nx, max_z, resolution);

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

} // namespace planning

} // namespace intel_wind

int main (int argc, char *argv[])
{
  return intel_wind::planning::plan_with_simple_setup();
}
