
#include <iostream>

#include <ompl/base/PlannerStatus.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/State.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/geometric/SimpleSetup.h>

#include "params.hpp"
#include "MyOptimizationObjective.hpp"
#include "MyProjection.hpp"
#include "MyRRTstar.hpp"
#include "MySE3StateSpace.hpp"

namespace intel_wind {

namespace planning {

namespace ob = ompl::base;
namespace og = ompl::geometric;

bool isStateValid(const ob::State *state) {
    return true;
}

int plan_with_simple_setup() {
  // construct the state space we are planning in
  ob::StateSpacePtr space(new MySE3StateSpace());

  // set the state space bounds
  ob::RealVectorBounds bounds(3);
  bounds.setLow(0, 0);
  bounds.setHigh(0, 1100);
  bounds.setLow(1, 0);
  bounds.setHigh(1, 1100);
  bounds.setLow(2, 0);
  bounds.setHigh(2, 600);
  space->as<MySE3StateSpace>()->setBounds(bounds);

  // set the default projection, not sure if it actually is required
  space->registerDefaultProjection(
      ob::ProjectionEvaluatorPtr(
          new MyProjection(space, proj_cellsize)));

  // configure the simple setup
  og::SimpleSetup ss(space);
  ss.setStateValidityChecker([](const ob::State *state) { return isStateValid(state); });

//  mission->getSSGSharedPtr()->setStateValidityChecker(
//      ob::StateValidityCheckerPtr(
//          new base::MyStateValidityCheckerClass<StateSpace>(mission, mission->getSSGSharedPtr()->getSpaceInformation())));
//
//  double res = *std::min_element(std::begin(mission->getAirplaneBB()), std::end(mission->getAirplaneBB()) - 1)
//      / (mission->getStateSpacePtr()->template as<StateSpace>()->getMaximumExtent()) * 0.5;
//  mission->getSSGSharedPtr()->getSpaceInformation()->setStateValidityCheckingResolution(res);  // 1%
//
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
      ss.simplifySolution();
      ss.getSolutionPath().print(std::cout);
  }
}

} // namespace planning

} // namespace intel_wind

int main (int argc, char *argv[])
{
  return intel_wind::planning::plan_with_simple_setup();
}
