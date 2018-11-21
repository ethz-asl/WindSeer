/*!
 * \file MyGoalSampleableRegion.hpp
 * \brief Contains class representing the goal
 *
 *  Created on: Nov 20, 2018
 *      Author: Florian Achermann, ASL
 */

#ifndef INTEL_WIND__MY_GOAL_SAMPLEABLE_REGION_HPP_
#define INTEL_WIND__MY_GOAL_SAMPLEABLE_REGION_HPP_

#include <vector>
#include <random>

#include <ompl/base/State.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/StateSpace.h>
#include <ompl/base/goals/GoalSampleableRegion.h>

#include "params.hpp"
#include "MySE3StateSpace.hpp"

namespace intel_wind {

namespace planning {

namespace ob = ompl::base;

/** \brief MyGoalSampleableRegion
 * Class to sample a goal from a goal region.
 * Currently no sampling is done and only the goal state is returned.
 */
class MyGoalSampleableRegion : public ob::GoalSampleableRegion {
 public:
  /** \brief Constructor */
  MyGoalSampleableRegion(const ob::SpaceInformationPtr& si_b,
                         const ob::StateSpacePtr& space);

  /** \brief sampleGoal
   * Returns exactly the goal state.
   */
  virtual void sampleGoal(ompl::base::State *st) const override;

  /** \brief maxSampleCount
   * Get the number of the maximum allowed sampled goal states.
   */
  virtual unsigned int maxSampleCount(void) const override;

  /** \brief setMaxSampleCount
   * Set the number of maximum allowed sampled goal states.
   */
  void setMaxSampleCount(unsigned int max_sample_count);

  /** \brief isSatisfied
   * Equivalent to calling isSatisfied(const State *, double *) with a NULL second argument.
   */
  virtual bool isSatisfied(const ob::State *st) const;

  /** \brief isSatisfied
   * Decide whether a given state is part of the
   * goal region. Returns true if the distance to goal is
   * less than the threshold (using distanceGoal())
   */
  virtual bool isSatisfied(const ob::State *st, double *distance) const override;

  /** \brief setGoalState
   * Set the goal state of the state space.
   */
  void setGoalState(const ob::ScopedState<MySE3StateSpace>& st);

  /** \brief getGoalState
   * Get the goal state of the corresponding state space.
   */
  const ob::State* getGoalState() const;

  /** \brief getGoalX
   * Get the x-position of the goal state of the corresponding state space.
   */
  double getGoalX() const;

  /** \brief getGoalY
   * Get the y-position of the goal state of the corresponding state space.
   */
  double getGoalY() const;

  /** \brief getGoalZ
   * Get the z-position of the goal state of the corresponding state space.
   */
  double getGoalZ() const;

  /** \brief printMyGoalSampleableRegion
   * Print the settings of MyGoalSampleableRegion
   */
  void printMyGoalSampleableRegion() const;

  /** \brief distanceGoal
   * Compute the distance from the input state to the goal state and return it.
   */
  virtual double distanceGoal(const ob::State *st) const override;

 private:
  /** \brief goal_
   * Goal State in the state space.
   */
  ob::ScopedState<MySE3StateSpace> goal_;

  /** \brief maxSampleCount
   * Maximum number of sampled goal states.
   */
  unsigned int maxSampleCount_;

  /** \brief rn_generator_
   * Random number generator which uses a mersenne twister engine.
   * http://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine
   */
  mutable std::mt19937 rn_generator_;

  /** \brief rn_generator_
   * Object to map the random number to the distribution desired.
   * http://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
   */
  mutable std::uniform_real_distribution<double> rn_distribution_;
};

} // namespace planning

} // namespace intel_wind

#endif /* INTEL_WIND__MY_GOAL_SAMPLEABLE_REGION_HPP_ */
