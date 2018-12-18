
#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <ompl/base/PlannerStatus.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/State.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/util/Console.h>

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

struct PredictionPlanningResult {
  std::string model_name = "";
  double planned_cost = 0.0;
  double execution_cost = 0.0;
};

struct PlanningResult {
  bool solution_possible = false;
  double reference_cost = 0.0;
  std::vector<PredictionPlanningResult> prediction_results;
};

struct SampleResult {
  std::string sample_name = "";
  std::vector<PlanningResult> planning_result;
};

SampleResult process_sample(const HDF5Interface::Sample& sample, std::vector<std::array<double, 8>> sg_configurations, double planning_time) {
  SampleResult result;
  result.sample_name = sample.sample_name;

  // setup the wind grid based on the reference wind field
  WindGridGeometry geo;
  geo.min_x = 0.0;
  geo.min_y = 0.0;
  geo.min_z = 0.0;
  geo.nx = sample.nx;
  geo.ny = sample.ny;
  geo.nz = sample.nz;
  geo.resolution_hor = sample.resolution_horizontal;
  geo.resolution_ver = sample.resolution_vertical;

  std::shared_ptr<WindGrid> wind_grid = std::make_shared<WindGrid>(WindGrid());
  std::shared_ptr<WindGrid> reference_wind_grid = std::make_shared<WindGrid>(WindGrid());
  reference_wind_grid->setWindGrid(sample.reference.wind_x, sample.reference.wind_y, sample.reference.wind_z, geo);

  // set terrain
  HeightMapClass map = HeightMapClass();
  map.setHeightMapData(sample.terrain, sample.nx, sample.nz * sample.resolution_vertical, sample.resolution_horizontal);

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
  ss.setOptimizationObjective(
          ob::OptimizationObjectivePtr(
                  new MyOptimizationObjective(ss.getSpaceInformation(), wind_grid, reference_wind_grid)));

  // set the planner
  MyRRTstar *my_rrtstar = new MyRRTstar(ss.getSpaceInformation());
  my_rrtstar->setRange(ss.getSpaceInformation()->getMaximumExtent() / 7.0);
  my_rrtstar->setGoalBias(0.05);
  my_rrtstar->setFocusSearch(false); // TODO occasionally get segfault when true, fix it
  // need to enable admissible cost to come, false sometimes leads to SEGFAULTs due to pruned goal states.
  my_rrtstar->setAdmissibleCostToCome(true);
  my_rrtstar->setCustomNearestNeighbor(false);
  my_rrtstar->setComputekNearestNeighbors2way(false);
  ss.setPlanner(ob::PlannerPtr(my_rrtstar));

  // define start and goal states
  ob::ScopedState<MySE3StateSpace> start(space);
  ob::ScopedState<MySE3StateSpace> goal(space);

  const int len = sg_configurations.size();
  int counter = 0;
  for(auto const sg_pair: sg_configurations) {
    PlanningResult planning_result;

    // clear the data from the previous run
    ss.clear();
    boost::static_pointer_cast<MyOptimizationObjective>(ss.getOptimizationObjective())->clear();

    // some info for the user
    counter++;
    std::cout << "\e[1m\tProcessing start/goal pair: " << counter << " out of " <<  len << "\e[0m" << std::endl;
    std::cout << "\e[1m\t\tPlanning with reference wind field\e[0m" << std::endl;

    // get the start and goal position TODO add also the bounding box so that it is at least valid with respect to z
    const double start_z = (map.getMaxZ() - map.getTerrainHeight(sg_pair[0], sg_pair[1]) - 0.5 * airplane_bb_param[2]) * sg_pair[2] +
            map.getTerrainHeight(sg_pair[0], sg_pair[1]) + 0.5 * airplane_bb_param[2];
    const double goal_z =  (map.getMaxZ() - map.getTerrainHeight(sg_pair[4], sg_pair[5]) - 0.5 * airplane_bb_param[2]) * sg_pair[6] +
            map.getTerrainHeight(sg_pair[4], sg_pair[5]) + 0.5 * airplane_bb_param[2];
    start->setXYZYaw(sg_pair[0], sg_pair[1], start_z, 0.0);
    goal->setXYZYaw(sg_pair[4], sg_pair[5], goal_z, 0.0);
    ss.setStartState(start);
    ss.setGoalState(goal);

    // plan with the reference wind field
    wind_grid->setWindGrid(sample.reference.wind_x, sample.reference.wind_y, sample.reference.wind_z, geo);
    ob::PlannerStatus solved = ss.solve(planning_time);

    // store the data in the planning result
    planning_result.solution_possible = solved == ob::PlannerStatus::EXACT_SOLUTION;
    planning_result.reference_cost =
            boost::static_pointer_cast<MyOptimizationObjective>(ss.getOptimizationObjective())->getCurrentBestCost();
    std::cout << "\e[1m\t\t\treference cost: " << planning_result.reference_cost << "\e[0m" << std::endl;

    if (planning_result.solution_possible) {
      // a solution with the cfd wind field is found, loop over all models
      for(auto const& prediction: sample.predictions) {
        PredictionPlanningResult prediction_result;
        prediction_result.model_name = prediction.model_name;

        std::cout << "\e[1m\t\tPlanning with the prediction from the '" << prediction.model_name << "' model\e[0m" << std::endl;
        ss.clear();
        boost::static_pointer_cast<MyOptimizationObjective>(ss.getOptimizationObjective())->clear();
        wind_grid->setWindGrid(prediction.wind_x, prediction.wind_y, prediction.wind_z, geo);

        solved = ss.solve(planning_time);

        prediction_result.planned_cost = boost::static_pointer_cast<MyOptimizationObjective>(ss.getOptimizationObjective())->getCurrentBestCost();

        // loop over solution path to determine if it is feasible and what the cost is
        boost::static_pointer_cast<MyOptimizationObjective>(ss.getOptimizationObjective())->setUseReference(true);

        double cost(0.0);
        og::PathGeometric path = ss.getSolutionPath();
        for (size_t idx_p = 1; idx_p < path.getStateCount(); ++idx_p) {
          const double intermediate_cost = ss.getOptimizationObjective()->motionCost(path.getState(idx_p - 1),path.getState(idx_p)).value();
          cost += intermediate_cost;
        }

        prediction_result.execution_cost = cost;
        std::cout << "\e[1m\t\t\tplanned cost:   " << prediction_result.planned_cost << "\e[0m" << std::endl;
        std::cout << "\e[1m\t\t\texecution cost: " << prediction_result.execution_cost << "\e[0m" << std::endl;

        boost::static_pointer_cast<MyOptimizationObjective>(ss.getOptimizationObjective())->setUseReference(false);

        planning_result.prediction_results.push_back(prediction_result);
      }
    }

    // add the planning results to the output
    result.planning_result.push_back(planning_result);
  }

  return result;
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

  // TODO: get the planning time from the arguments
  const double planning_time = 2.0;

  SampleResult results[database.getNumberSamples()];
  // TODO parallelize this loop
  for (int i = 0; i < database.getNumberSamples(); ++i) {
    // change to \e[1;7m for inverse colors
    std::cout << "\e[1mStart processing sample: " << i+1 << " out of " << database.getNumberSamples() << " (" << database.getSample(i).sample_name << ")\e[0m" << std::endl;
    results[i] = process_sample(database.getSample(i), sg_configurations, planning_time);
  }

  // TODO store the results in an hdf5 database

  return 0;
}

} // namespace planning

} // namespace intel_wind

int main (int argc, char *argv[]) {
  ompl::msg::setLogLevel(ompl::msg::LOG_WARN);
  return intel_wind::planning::benchmark(argc, argv);
}
