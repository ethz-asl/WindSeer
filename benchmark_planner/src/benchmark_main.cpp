#include <algorithm>
#include <array>
#include <fstream>
#include <H5Cpp.h>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
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


char* getCmdOption(char ** begin, char ** end, const std::string & option) {
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}


bool cmdOptionExists(char** begin, char** end, const std::string& option) {
  return std::find(begin, end, option) != end;
}


bool file_exists(const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}


SampleResult process_sample(const HDF5Interface::Sample& sample,
                            std::vector<std::array<double, 8>> sg_configurations,
                            double planning_time,
                            double reference_multiplier) {
  // decide if the info should be printed if OpenMP is available
  const char* omp_num_threads =  std::getenv("OMP_NUM_THREADS");
  bool print_info(false);
  if (omp_num_threads != NULL) {
    std::stringstream s;
    s << omp_num_threads;
    int val;
    s >> val;
    print_info = val == 1;
  }

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
  my_rrtstar->setFocusSearch(true); // TODO occasionally get segfault when true, fix it
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

#if not defined(_OPENMP)
    std::cout << "\e[1m\tProcessing start/goal pair: " << counter << " out of " <<  len << "\e[0m" << std::endl;
    std::cout << "\e[1m\t\tPlanning with reference wind field\e[0m" << std::endl;
#else
    if (print_info) {
      std::cout << "\e[1m\tProcessing start/goal pair: " << counter << " out of " <<  len << "\e[0m" << std::endl;
      std::cout << "\e[1m\t\tPlanning with reference wind field\e[0m" << std::endl;
    }
#endif

    // get the start and goal position TODO add also the bounding box so that it is at least valid with respect to z
    const double start_z = (map.getMaxZ() - map.getTerrainHeight(sg_pair[0], sg_pair[1]) - airplane_bb_param[2]) * sg_pair[2] +
            map.getTerrainHeight(sg_pair[0], sg_pair[1]) + airplane_bb_param[2];
    const double goal_z =  (map.getMaxZ() - map.getTerrainHeight(sg_pair[4], sg_pair[5]) - airplane_bb_param[2]) * sg_pair[6] +
            map.getTerrainHeight(sg_pair[4], sg_pair[5]) + airplane_bb_param[2];
    start->setXYZYaw(sg_pair[0], sg_pair[1], start_z, 0.0);
    goal->setXYZYaw(sg_pair[4], sg_pair[5], goal_z, 0.0);
    ss.setStartState(start);
    ss.setGoalState(goal);

    // plan with the reference wind field
    wind_grid->setWindGrid(sample.reference.wind_x, sample.reference.wind_y, sample.reference.wind_z, geo);
    ob::PlannerStatus solved = ss.solve(planning_time * reference_multiplier);

    // store the data in the planning result
    planning_result.solution_possible = solved == ob::PlannerStatus::EXACT_SOLUTION;
    planning_result.reference_cost =
            boost::static_pointer_cast<MyOptimizationObjective>(ss.getOptimizationObjective())->getCurrentBestCost();

#if not defined(_OPENMP)
    std::cout << "\e[1m\t\t\treference cost: " << planning_result.reference_cost << "\e[0m" << std::endl;
#else
    if (print_info) {
      std::cout << "\e[1m\t\t\treference cost: " << planning_result.reference_cost << "\e[0m" << std::endl;
    }
#endif

    // a solution with the cfd wind field is found, loop over all models
    for(auto const& prediction: sample.predictions) {
      PredictionPlanningResult prediction_result;
      prediction_result.model_name = prediction.model_name;

#if not defined(_OPENMP)
      std::cout << "\e[1m\t\tPlanning with the prediction from the '" << prediction.model_name << "' model\e[0m" << std::endl;
#else
      if (print_info) {
        std::cout << "\e[1m\t\tPlanning with the prediction from the '" << prediction.model_name << "' model\e[0m" << std::endl;
      }
#endif

      ss.clear();
      boost::static_pointer_cast<MyOptimizationObjective>(ss.getOptimizationObjective())->clear();
      wind_grid->setWindGrid(prediction.wind_x, prediction.wind_y, prediction.wind_z, geo);

      solved = ss.solve(planning_time);

      prediction_result.planned_cost = boost::static_pointer_cast<MyOptimizationObjective>(ss.getOptimizationObjective())->getCurrentBestCost();

      if (solved == ob::PlannerStatus::EXACT_SOLUTION) {
        // loop over solution path to determine if it is feasible and what the cost is
        boost::static_pointer_cast<MyOptimizationObjective>(ss.getOptimizationObjective())->setUseReference(true);

        double cost(0.0);
        og::PathGeometric path = ss.getSolutionPath();
        for (size_t idx_p = 1; idx_p < path.getStateCount(); ++idx_p) {
          const double intermediate_cost = ss.getOptimizationObjective()->motionCost(path.getState(idx_p - 1),path.getState(idx_p)).value();
          cost += intermediate_cost;
        }

        prediction_result.execution_cost = cost;

#if not defined(_OPENMP)
        std::cout << "\e[1m\t\t\tplanned cost:   " << prediction_result.planned_cost << "\e[0m" << std::endl;
        std::cout << "\e[1m\t\t\texecution cost: " << prediction_result.execution_cost << "\e[0m" << std::endl;
#else
        if (print_info) {
          std::cout << "\e[1m\t\t\tplanned cost:   " << prediction_result.planned_cost << "\e[0m" << std::endl;
          std::cout << "\e[1m\t\t\texecution cost: " << prediction_result.execution_cost << "\e[0m" << std::endl;
        }
#endif

        boost::static_pointer_cast<MyOptimizationObjective>(ss.getOptimizationObjective())->setUseReference(false);
      } else {
        prediction_result.execution_cost = std::numeric_limits<double>::infinity();
        prediction_result.planned_cost = std::numeric_limits<double>::infinity();
#if not defined(_OPENMP)
        std::cout << "\e[1m\t\t\tno solution found\e[0m" << std::endl;
#else
        if (print_info) {
          std::cout << "\e[1m\t\t\tno solution found\e[0m" << std::endl;
        }
#endif
      }

      planning_result.prediction_results.push_back(prediction_result);
    }

    // add the planning results to the output
    result.planning_result.push_back(planning_result);
  }

  return result;
}


void write_hdf5_database(SampleResult results[], int size, std::string database_name) {
  // create the HDF5 file and the necessary dataspaces
  H5::H5File file = H5::H5File(database_name, H5F_ACC_TRUNC);
  H5::DataSet dataset;

  hsize_t dimsf[1];
  dimsf[0] = 1;
  H5::DataSpace dataspace(1, dimsf);

  H5::IntType datatype_bool(H5::PredType::NATIVE_INT);
  H5::FloatType datatype_double(H5::PredType::NATIVE_DOUBLE);

  // loop over samples
  for (int i = 0; i < size; ++i) {
    H5::Group sample_group = H5::Group(file.createGroup(results[i].sample_name.c_str()));

    // loop over start and goal configurations
    int counter = 0;
    for (auto const& planning_result: results[i].planning_result) {
      // only write data if a path was possible
      std::ostringstream oss;
      oss << "configuration" << counter;
      H5::Group configuration_group = H5::Group(sample_group.createGroup(oss.str().c_str()));

      dataset = configuration_group.createDataSet("path_possible", datatype_bool, dataspace);
      const int path_possible = planning_result.solution_possible;
      dataset.write(&path_possible, H5::PredType::NATIVE_INT);

      dataset = configuration_group.createDataSet("reference_cost", datatype_double, dataspace);
      dataset.write(&planning_result.reference_cost, H5::PredType::NATIVE_DOUBLE);

      H5::Group prediction_main_group = H5::Group(configuration_group.createGroup("predictions"));
      for (auto const& prediction_result: planning_result.prediction_results) {
        H5::Group prediction_group = H5::Group(prediction_main_group.createGroup(prediction_result.model_name.c_str()));

        dataset = prediction_group.createDataSet("planned_cost", datatype_double, dataspace);
        dataset.write(&prediction_result.planned_cost, H5::PredType::NATIVE_DOUBLE);

        dataset = prediction_group.createDataSet("execution_cost", datatype_double, dataspace);
        dataset.write(&prediction_result.execution_cost, H5::PredType::NATIVE_DOUBLE);
      }

      ++counter;
    }
  }
}


int benchmark(int argc, char *argv[]) {
  // parse the input arguments
  if(cmdOptionExists(argv, argv+argc, "-h") || cmdOptionExists(argv, argv+argc, "--help")) {
    std::cout << "usage: ./benchmark [-h][-i INPUT_DATASET]" << std::endl;
    std::cout << "                   [-o OUTPUT_DATASET]" << std::endl;
    std::cout << "                   [-t PLANNING_TIME]" << std::endl;
    std::cout << "                   [-rm REFERENCE_MULTIPLIER]" << std::endl;
    std::cout << std::endl;
    std::cout << "Planning benchmark for the wind prediction models" << std::endl;
    std::cout << std::endl;
    std::cout << "optional arguments" << std::endl;
    std::cout << "  -h, --help\t\t\tshow this message and exit" << std::endl;
    std::cout << "  -i INPUT_DATASET\t\tfilename of the input database (default prediction.hdf5)" << std::endl;
    std::cout << "  -o OUTPUT_DATASET\t\tfilename of the output database (default planning_results.hdf5)" << std::endl;
    std::cout << "  -t PLANNING_TIME\t\tplanning time for each planning case [s] (default 10)" << std::endl;
    std::cout << "  -rm REFERENCE_MULTIPLIER\tfactor for the planning time with the reference wind field [-] (default 2)" << std::endl;

    return 0;
  }

  char* value = getCmdOption(argv, argv + argc, "-i");
  std::string input_database_name = "prediction.hdf5";
  if (value) {
    input_database_name = std::string(value);
  }

  value = getCmdOption(argv, argv + argc, "-o");
  std::string output_database_name = "planning_results.hdf5";
  if (value) {
    output_database_name = std::string(value);
  }

  value = getCmdOption(argv, argv + argc, "-t");
  double planning_time = 10.0;
  if (value) {
    planning_time = atof(value);

    if (planning_time < 0.0) {
      throw std::invalid_argument("The planning time needs to be positive: " + std::to_string(planning_time));
    }
  }

  value = getCmdOption(argv, argv + argc, "-rm");
  double reference_multiplier = 2;
  if (value) {
    reference_multiplier = atof(value);

    if (reference_multiplier < 1.0) {
      throw std::invalid_argument("The reference multiplier needs to be larger than 1.0: " + std::to_string(reference_multiplier));
    }
  }

  value = getCmdOption(argv, argv + argc, "-c");
  std::string configurations_file_name = "start_goal_configurations.txt";
  if (value) {
      configurations_file_name = std::string(value);
  }

  // check if the input files exist
  if (!file_exists(input_database_name)) {
    throw std::invalid_argument("The specified input database does not exist: " + input_database_name);
  }

  if (!file_exists(configurations_file_name)) {
    throw std::invalid_argument("The file for the start/goal configurations does not exist: " + configurations_file_name);
  }

  // open the specified file and the specified dataset in the file.
  HDF5Interface database;
  database.init(input_database_name);

  // open the file with the start and goal configurations
  std::ifstream sg_file(configurations_file_name);

  // read the configurations from file
  std::vector<std::array<double, 8>> sg_configurations = {};
  std::array<double, 8> tmp = {};
  while (sg_file >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3] >> tmp[4] >> tmp[5] >> tmp[6] >> tmp[7]) {
    sg_configurations.push_back(tmp);
  }
  sg_file.close();

  SampleResult results[database.getNumberSamples()];
  int i;

#pragma omp parallel for
  for (i = 0; i < database.getNumberSamples(); ++i) {
    HDF5Interface::Sample sample;
#pragma omp critical
    {
    sample = database.getSample(i);
    std::cout << "\e[1mStart processing sample: " << i+1 << " out of " << database.getNumberSamples() << " (" << sample.sample_name << ")\e[0m" << std::endl;
    }

    // change to \e[1;7m for inverse colors
    SampleResult tmp = process_sample(sample, sg_configurations, planning_time, reference_multiplier);
#pragma omp critical
    results[i] = tmp;
  }

  // write the results into a hdf5 database
  write_hdf5_database(results, database.getNumberSamples(), output_database_name);

  return 0;
}

} // namespace planning

} // namespace intel_wind

int main (int argc, char *argv[]) {
  ompl::msg::setLogLevel(ompl::msg::LOG_WARN);
  return intel_wind::planning::benchmark(argc, argv);
}
