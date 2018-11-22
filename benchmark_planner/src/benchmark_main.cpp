
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

namespace intel_wind {

namespace planning {

namespace ob = ompl::base;
namespace og = ompl::geometric;

int plan_with_simple_setup() {
  // set the terrain
  const int nx = 64;
  const int ny = 64;
  const double resolution = 17.1875;
  const double max_z = 1100.0 / 96.0 * 64.0;

  std::mt19937 generator(std::random_device{}());
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
  ss.setOptimizationObjective(ob::OptimizationObjectivePtr(new MyOptimizationObjective(ss.getSpaceInformation())));

  // set the planner
  MyRRTstar *my_rrtstar = new MyRRTstar(ss.getSpaceInformation());
  my_rrtstar->setRange(ss.getSpaceInformation()->getMaximumExtent() / 7.0);
  my_rrtstar->setGoalBias(0.05);
  my_rrtstar->setFocusSearch(false);
  // need to enable admissible cost to come, false sometimes leads to SEGFAULTs due to pruned goal states.
  my_rrtstar->setAdmissibleCostToCome(true);
  my_rrtstar->setCustomNearestNeighbor(true);
  my_rrtstar->setComputekNearestNeighbors2way(true);
  ss.setPlanner(ob::PlannerPtr(my_rrtstar));

  // define start and goal
  ob::ScopedState<> start(space);
  start.random();

  ob::ScopedState<> goal(space);
  goal.random();

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
