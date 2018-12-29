/*!
 * 	MyOptimizationObjective.cpp
 *
 *  Created on: Nov, 21 2018
 *      Author: Florian Achermann, ASL
 */

#include "MyOptimizationObjective.hpp"

#include <limits>
#include <assert.h>
#include <iostream>

#include <boost/make_shared.hpp>
#include <ompl/config.h>

#include <ompl/base/goals/GoalRegion.h>

#include "params.hpp"
#include "MySampler.hpp"
#include "MySE3StateSpace.hpp"
#include "MyGoalSampleableRegion.hpp"

namespace intel_wind {

namespace planning {

MyOptimizationObjective::MyOptimizationObjective(const ob::SpaceInformationPtr& si,
                                                 std::shared_ptr<WindGrid> wind_grid,
                                                 std::shared_ptr<WindGrid> reference_wind_grid)
        : ob::PathLengthOptimizationObjective(si),
          wind_grid_(wind_grid),
          reference_wind_grid_(reference_wind_grid) {
  setCostToGoHeuristic(boost::bind(&MyOptimizationObjective::goalRegionTimeToGo, this, _1, _2));
}


MyOptimizationObjective::~MyOptimizationObjective() {
}


bool MyOptimizationObjective::isSatisfied(ob::Cost c) const {
  bool update = false;

  // check if the current solution is the initial solution
  if (currentBestCost_ > c.value()) {
      currentBestCost_ = c.value();
    if (not hasSolution_) {
      hasSolution_ = true;

      if (verbose_)
        std::cout << "MyOptimizationObjective: Found initial path with cost: " << c.value() << cost_unit_ << std::endl;
    } else {
      if (verbose_)
        std::cout << "MyOptimizationObjective: Found new path with lower cost: " << c.value() << cost_unit_ << std::endl;
    }
  }

  return false;
}


ob::Cost MyOptimizationObjective::motionCostHeuristic(const ob::State *s1, const ob::State *s2) const {
  const double path_length = si_->getStateSpace()->as<MySE3StateSpace>()->euclidean_distance(s1, s2);

  return ob::Cost(path_length / (computeProjectedWindMagnitude(s1, s2) + v_air_param));
}


ob::Cost MyOptimizationObjective::motionCost(const ob::State *s1, const ob::State *s2) const {
  return ob::Cost(computeEuclideanTimeoptimalCost(s1, s2));
}


bool MyOptimizationObjective::isCostBetterThan(ob::Cost c1, ob::Cost c2) const {
  if (std::isnan(c2.value()))
    if (not std::isnan(c1.value()))
      return true;
  return c1.value() < c2.value();
}


void MyOptimizationObjective::clear() {
  currentBestCost_ = std::numeric_limits<double>::infinity();
  hasSolution_ = false;
}


void MyOptimizationObjective::setUseReference(const bool use_reference) {
  useReference_ = use_reference;
}


bool MyOptimizationObjective::getHasSolution() const {
  return hasSolution_;
}


double MyOptimizationObjective::getCurrentBestCost() const {
  return currentBestCost_;
}


void MyOptimizationObjective::setVerbose(bool verbose) {
  verbose_ = verbose;
}


double MyOptimizationObjective::computeProjectedWindMagnitude(const ob::State *s1, const ob::State *s2) const {
  const double dx = s2->as<typename MySE3StateSpace::StateType>()->getX() - s1->as<typename MySE3StateSpace::StateType>()->getX();
  const double dy = s2->as<typename MySE3StateSpace::StateType>()->getY() - s1->as<typename MySE3StateSpace::StateType>()->getY();
  const double dz = s2->as<typename MySE3StateSpace::StateType>()->getZ() - s1->as<typename MySE3StateSpace::StateType>()->getZ();
  const double eucl_dist = sqrt(dx*dx + dy*dy + dz*dz);

  if (eucl_dist <= 0.0)
    return 0.0;

  if (useReference_) {
    return fabs(dx)/eucl_dist * reference_wind_grid_->getMaxWindMagnitudeX() +
           fabs(dy)/eucl_dist * reference_wind_grid_->getMaxWindMagnitudeY() +
           fabs(dz)/eucl_dist * reference_wind_grid_->getMaxWindMagnitudeZ();
  } else {
    return fabs(dx)/eucl_dist * wind_grid_->getMaxWindMagnitudeX() +
           fabs(dy)/eucl_dist * wind_grid_->getMaxWindMagnitudeY() +
           fabs(dz)/eucl_dist * wind_grid_->getMaxWindMagnitudeZ();
  }
}


ob::InformedSamplerPtr MyOptimizationObjective::allocInformedStateSampler(const ob::ProblemDefinitionPtr probDefn,
                                                                                      unsigned int maxNumberCalls) const {

  std::cout << "allocating new state sampler" << std::endl;
  // Make the direct path-length informed sampler and return. If OMPL was compiled with Eigen, a direct version is available, if not a rejection-based technique can be used
#if OMPL_HAVE_EIGEN3
  if (useReference_) {
    return boost::make_shared<MySampler>(probDefn, maxNumberCalls,
            reference_wind_grid_->getMaxWindMagnitudeX(), reference_wind_grid_->getMaxWindMagnitudeY(), reference_wind_grid_->getMaxWindMagnitudeZ());
  } else {
    return boost::make_shared<MySampler>(probDefn, maxNumberCalls,
            wind_grid_->getMaxWindMagnitudeX(), wind_grid_->getMaxWindMagnitudeY(), wind_grid_->getMaxWindMagnitudeZ());
  }


#else
  throw Exception("Direct sampling of the path-length objective requires Eigen, but this version of OMPL was compiled without Eigen support. If possible, please install Eigen and recompile OMPL. If this is not possible, you can manually create an instantiation of RejectionInfSampler to approximate the behaviour of direct informed sampling.");
  // Return a null pointer to avoid compiler warnings
  return ompl::base::InformedSamplerPtr();
#endif
}


ob::Cost MyOptimizationObjective::goalRegionTimeToGo(const ob::State *state, const ob::Goal *goal) {
  const ob::GoalRegion *goalRegion = goal->as<ob::GoalRegion>();

  // Ensures that all states within the goal region's threshold to
  // have a cost-to-go of exactly zero.
  return ob::Cost(std::max(goalRegion->distanceGoal(state) - goalRegion->getThreshold(),
                             0.0) / (v_air_param +
                                 computeProjectedWindMagnitude(state,
                                     goal->as<MyGoalSampleableRegion>()->getGoalState())));

}


double MyOptimizationObjective::computeEuclideanTimeoptimalCost(const ob::State *s1, const ob::State *s2) const {
  const double dist = si_->distance(s1, s2);
  if (dist < 1e-5) {
      return 0.0;
  }

  const double vair_inv = 1.0 / v_air_param;
  const double distance_inv = 1.0 / dist;
  const double dx = s2->as<typename MySE3StateSpace::StateType>()->getX() - s1->as<typename MySE3StateSpace::StateType>()->getX();
  const double dy = s2->as<typename MySE3StateSpace::StateType>()->getY() - s1->as<typename MySE3StateSpace::StateType>()->getY();
  const double dz = s2->as<typename MySE3StateSpace::StateType>()->getZ() - s1->as<typename MySE3StateSpace::StateType>()->getZ();
  double dt_max = wind_grid_->getResolution() * 0.5 * distance_inv;
  if (useReference_) {
    dt_max = reference_wind_grid_->getResolution() * 0.5 * distance_inv;
  }

  double t_interpol(0.0), dt(0.0), wind_normal(0.0), wind_forward(0.0), airspeed_forward(0.0), cost(0.0);
  float u(0.0f), v(0.0f), w(0.0f);

  ob::State *state_interpol = si_->allocState();

  while (t_interpol < 1.0) {
    // get dt for this interpolation
    dt = std::min(dt_max, 1.0 - t_interpol);

    si_->getStateSpace()->interpolate(s1, s2, t_interpol, state_interpol);

    if (useReference_) {
      reference_wind_grid_->getWind(
                   state_interpol->as<MySE3StateSpace::StateType>()->getX(),
                   state_interpol->as<MySE3StateSpace::StateType>()->getY(),
                   state_interpol->as<MySE3StateSpace::StateType>()->getZ(),
                   &u, &v, &w);
    } else {
      wind_grid_->getWind(
                   state_interpol->as<MySE3StateSpace::StateType>()->getX(),
                   state_interpol->as<MySE3StateSpace::StateType>()->getY(),
                   state_interpol->as<MySE3StateSpace::StateType>()->getZ(),
                   &u, &v, &w);
    }


    wind_forward = (dx * u + dy * v + dz * w) * distance_inv;
    wind_normal = sqrt(u * u + v * v + w * w - wind_forward * wind_forward);

    double scale = 0.8; // safety margin of 20 %
    if (useReference_) {
        scale = 1.0;
    }

    // check for too strong vertical wind
    if (fabsf(w) > v_air_param * sinf(max_pitch) * scale) {
      si_->freeState(state_interpol);
      return std::numeric_limits<double>::infinity();
    }

    // check if normal wind exceeds airspeed
    if ((wind_normal > v_air_param * scale) || (std::isnan(wind_normal))) {
      si_->freeState(state_interpol);
      return std::numeric_limits<double>::infinity();
    }

    airspeed_forward = sqrt(v_air_param * v_air_param - wind_normal * wind_normal);

    // check if the airplane is able to move forward after correcting for the normal wind
    if (0.0 > airspeed_forward + wind_forward / scale) {
      si_->freeState(state_interpol);
      return std::numeric_limits<double>::infinity();
    }

    cost += dt * dist / (airspeed_forward + wind_forward);

    // update the interpolation time
    t_interpol += dt;
  }

  si_->freeState(state_interpol);

  if (cost < 0.0) {
    std::cout << "negative cost, should never happen" << std::endl;
    return std::numeric_limits<double>::infinity();
  }

  if (std::isnan(cost)) {
    std::cout << "nan cost, should never happen" << std::endl;
    return std::numeric_limits<double>::infinity();
  }

  return cost;
}

} // namespace planning

} // namespace intel_wind
