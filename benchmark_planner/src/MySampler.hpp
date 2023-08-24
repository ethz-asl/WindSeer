/*!
 *  \file MySampler.hpp
 *  Allows sampling from an ellipsoid and also implements Dubins airplane adapted informed sampling.
 *
 *  Created on: Nov, 21 2018
 *      Author: Florian Achermann, ASL
 *  Copy from OMPL (http://ompl.kavrakilab.org/classompl_1_1base_1_1PathLengthDirectInfSampler.html)!
 */

#ifndef INTEL_WIND__MY_SAMPLER_HPP_
#define INTEL_WIND__MY_SAMPLER_HPP_

#include <vector>
#include <list>

#include <boost/shared_ptr.hpp>
#include <ompl/base/State.h>
#include <ompl/base/Cost.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/samplers/InformedStateSampler.h>
#include <ompl/base/StateSampler.h>
#include <ompl/util/ProlateHyperspheroid.h>
#include <ompl/util/RandomNumbers.h>
#include <ompl/config.h>

#if !OMPL_HAVE_EIGEN3
#error The MySampler class uses Eigen3, which was not detected at build time.
#endif


namespace intel_wind {

namespace planning {

namespace ob = ompl::base;

/** \brief An informed sampler for problems seeking to minimize path length.*/
class MySampler : public ob::InformedSampler {
private:
  /** \brief N_SHIFT
   * If a sample is determined invalid and shifted this determines the increment in z direction in the following way:
   * dz = InformedSampler::space_->getLongestValidSegmentLength() * N_SHIFT
   *
   * N_SHIFT is unitless (multiple of longestValidSegmentLength)
   *
   * The value must be bigger than 0!!
   */
  const int N_SHIFT = 10;
public:

  /** \brief Construct a sampler that only generates states with a heuristic solution estimate that is less than the cost of the current solution using a direct ellipsoidal method. */
  MySampler(const ob::ProblemDefinitionPtr probDefn, unsigned int maxNumberCalls, double max_wind_x, double max_wind_y, double max_wind_z);

  virtual ~MySampler();

  /** \brief Sample uniformly in the subset of the state space whose heuristic solution estimates are less than the provided cost, i.e. in the interval [0, maxCost). Returns false if such a state was not found in the specified number of iterations. */
  virtual bool sampleUniform(ob::State *statePtr, const ob::Cost &maxCost);

  /** \brief Sample uniformly in the subset of the state space whose heuristic solution estimates are between the provided costs, [minCost, maxCost). Returns false if such a state was not found in the specified number of iterations. */
  virtual bool sampleUniform(ob::State *statePtr, const ob::Cost &minCost, const ob::Cost &maxCost);

  /** \brief Whether the sampler can provide a measure of the informed subset */
  virtual bool hasInformedMeasure() const;

  /** \brief The measure of the subset of the state space defined by the current solution cost that is being searched. Does not consider problem boundaries but returns the measure of the entire space if no solution has been found. In the case of multiple goals, this measure assume each individual subset is independent, therefore the resulting measure will be an overestimate if any of the subsets overlap. */
  virtual double getInformedMeasure(const ob::Cost &currentCost) const;

  /** \brief A helper function to calculate the heuristic estimate of the solution cost for the informed subset of a given state. */
  virtual ob::Cost heuristicSolnCost(const ob::State *statePtr) const;

private:
  // Helper functions:
  // High level
  /** \brief Sample uniformly in the subset of the state space whose heuristic solution estimates are less than the provided cost using a persistent iteration counter */
  bool sampleUniform(ob::State *statePtr, const ob::Cost &maxCost, unsigned int *iters);

  /** \brief Sample from the bounds of the problem and keep the sample if it passes the given test. Meant to be used with isInAnyPhs and phsPtr->isInPhs() */
  bool sampleBoundsRejectPhs(ob::State* statePtr, unsigned int *iters, const ob::Cost &maxCost);

  /** \brief Sample from the given PHS and return true if the sample is within the boundaries of the problem (i.e., it \e may be kept). */
  bool samplePhsRejectBounds(ob::State *statePtr, unsigned int *iters, const ob::Cost &maxCost);

  // Low level
  /** \brief Extract the informed subspace from a state pointer */
  std::vector<double> getInformedSubstate(const ob::State *statePtr) const;

  /** \brief Create a full vector with any uninformed subspaces filled with a uniform random state. Expects the state* to be allocated */
  void createFullState(ob::State * statePtr, const std::vector<double> &informedVector);

  /** \brief Iterate through the list of PHSs and update each one to the new max cost as well as the sum of their measures. Will remove any PHSs that cannot improve a better solution and in that case update the number of goals. */
  void updatePhsDefinition(const ob::Cost &maxCost);

  /** \brief Shift the sample up in z to get a collision free/valid sample. */
  bool getValidSampleByUpshifting(ob::State *statePtr, const ob::Cost &maxCost);

  // Variables

  /** \brief Start state. */
  std::vector<double> startState_;

  /** \brief Start state. */
  std::vector<double> goalState_;

  /** \brief The prolate hyperspheroid description of the sub problems. One per goal state. */
  ompl::ProlateHyperspheroidPtr phsPtr_;

  /** \brief The summed measure of all the start-goal pairs */
  double summedMeasure_;

  /** \brief The index of the subspace of a compound StateSpace for which we can do informed sampling. Unused if the StateSpace is not compound. */
  unsigned int informedIdx_;

  /** \brief The state space of the planning problem that is informed by the heuristics, i.e., in SE(2), R^2*/
  ob::StateSpacePtr informedSubSpace_;

  /** \brief The index of the subspace of a compound StateSpace for which we cannot do informed sampling. Unused if the StateSpace is not compound. */
  unsigned int uninformedIdx_;

  /** \brief The state space of the planning problem that is \e not informed by the heuristics, i.e., in SE(2), SO(2)*/
  ob::StateSpacePtr uninformedSubSpace_;

  /** \brief A regular sampler for the entire statespace for cases where informed sampling cannot be used or is not helpful. I.e., Before a solution is found, or if the solution does not reduce the search space. */
  ob::StateSamplerPtr baseSampler_;

  /** \brief A regular sampler to use on the uninformed subspace. */
  ob::StateSamplerPtr uninformedSubSampler_;

  /** \brief An instance of a random number generator */
  ompl::RNG rng_;

  /** \brief A pointer to the state validity checker */
  ob::StateValidityCheckerPtr stateValidityChecker_;

  /** \brief The maximum z value of all goals and starts positions */
  double zMaxStartGoal_;

  /** \brief The minimum z value of all goals and starts positions */
  double zMinStartGoal_;

  /** \brief Indicates if obstacle aware sampling should be executed. */
  bool obstacleAwareSampling_;

  /** \brief Upper bound of the state space in z direction. */
  double stateSpaceZMax_;

  /** \brief Maximum speed of the airplane, airspeed combined with the maximum wind speed. */
  double maxSpeed_;

  /** \brief Maximum magnitude of the wind in x-direction. */
  double maxWindX_;

  /** \brief Maximum magnitude of the wind in y-direction. */
  double maxWindY_;

  /** \brief Maximum magnitude of the wind in z-direction. */
  double maxWindZ_;
};

} // namespace planning

} // namespace intel_wind

#endif // INTEL_WIND__MY_SAMPLER_HPP_
