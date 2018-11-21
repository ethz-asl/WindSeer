/*!
 * 	\file MyOptimizationObjective.hpp
 *
 *  Created on: Nov, 20 2018
 *      Author: Daniel Schneider, ASL
 *              Florian Achermann, ASL
 */

#ifndef INTEL_WIND__MY_OPTIMIZATION_OBJECTIVE_HPP_
#define INTEL_WIND__MY_OPTIMIZATION_OBJECTIVE_HPP_


#include <deque>
#include <ratio>
#include <chrono>
#include <utility>
#include <string>
#include <limits>

#include <boost/shared_ptr.hpp>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/State.h>
#include <ompl/base/Goal.h>
#include <ompl/base/Cost.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/samplers/InformedStateSampler.h>

// TODO write meteo grid class

namespace intel_wind {

namespace planning {

namespace ob = ompl::base;

/** \brief MyOptimizationObjective
 * Class which contains an optimization objective. The current objective is the path length.
 *
 * TODO Handling with lastSolutions_ could be improved, for example leave it empty and check if it is empty to check if a solution exists
 */
class MyOptimizationObjective : public ob::PathLengthOptimizationObjective {
 public:
  /** \brief Constructor */
  MyOptimizationObjective(const ob::SpaceInformationPtr &si);

  /** \brief Destructor. */
  virtual ~MyOptimizationObjective() override;

  /** \brief isSatisfied
   * Check if the input cost satisfies any of the finishing criteria.
   */
  virtual bool isSatisfied(ob::Cost c) const override;

  /** \brief motionCostHeuristic
   * Compute a heuristic which (under)estimates the actual cost. Is used for BITstar. Does not work with distance defined in DubinsAirplane2.cpp.
   *
   * TODO: Check if the calling of distance instead of euclidean_distance justifies the additional computations.
   */
  virtual ob::Cost motionCostHeuristic(const ob::State *s1, const ob::State *s2) const override;

  /** \brief motionCost
   * Computes the cost of a motion from s1 to s2.
   */
  virtual ob::Cost motionCost(const ob::State *s1, const ob::State *s2) const override;

  /** \brief isCostBetterThan
   * Check whether the the cost \e c1 is considered better than the cost \e c2.
   */
  virtual bool isCostBetterThan(ob::Cost c1, ob::Cost c2) const override;

 private:
  /** \brief computeProjectedWindMagnitude
   * Return the magnitude of the maximum wind projected in the direction of the euclidean path.
   */
  double computeProjectedWindMagnitude(const ob::State *s1, const ob::State *s2) const;

  /** \brief allocInformedStateSampler
   * Allocate a state sampler for path length objective (i.e., direct ellipsoidal sampling and rejecting if
   * not satisfying bounds in z direction).
   */
  virtual ob::InformedSamplerPtr allocInformedStateSampler(const ob::ProblemDefinitionPtr probDefn,
                                                           unsigned int maxNumberCalls) const;

  /** \brief goalRegionTimeToGo
   * For use when the cost-to-go of a state under the optimization objective is equivalent to the
   * minimum time to get there. This function assumes that all states within the goal region's
   * threshold have a cost-to-go of exactly zero.
   * Note: \e goal is assumed to be of type ompl::base::GoalRegion
   */
  ob::Cost goalRegionTimeToGo(const ob::State *state, const ob::Goal *goal);

  /** \brief computeEuclideanTimeoptimalCost
   * Compute the time it takes to fly a straight line in wind.
   */
  double computeEuclideanTimeoptimalCost(const ob::State *s1, const ob::State *s2) const;

  /** \brief isSymmetric
   * The cost is not symmetric
   */
  bool isSymmetric() const override {
      return false;
  }

  /** \brief String for the unit of the cost. */
  std::string cost_unit_ = " s";


  /** \brief Cache if a solution is known. */
  mutable bool hasSolution_ = false;

  mutable double currentBestCost_ = std::numeric_limits<double>::infinity();
};

} // namespace planning

} // namespace intel_wind

#endif /* INTEL_WIND__MY_OPTIMIZATION_OBJECTIVE_HPP_ */
