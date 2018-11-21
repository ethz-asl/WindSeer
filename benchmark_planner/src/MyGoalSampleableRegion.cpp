/*!
 * \file MyGoalSampleableRegion.cpp
 *
 *  Created on: Nov 20, 2018
 *      Author: Florian Achermann, ASL
 */

#include "MyGoalSampleableRegion.hpp"

#include <iostream>
#include <cmath>

namespace intel_wind {

namespace planning {

MyGoalSampleableRegion::MyGoalSampleableRegion(const ob::SpaceInformationPtr& si_b,
                                                           const ob::StateSpacePtr& space)
    : /* Note for planningApproach_ == 3:
       * Space information is set to m_ssgc simple setup which uses the Dubins distance. */
       ob::GoalSampleableRegion(si_b),
       goal_(space),
       maxSampleCount_(max_num_goal_states),
       rn_generator_(std::random_device{}()),
       rn_distribution_(-M_PI, M_PI) {
  setThreshold(goal_threshold);
}


void MyGoalSampleableRegion::sampleGoal(ompl::base::State *st) const {
  st->as<typename MySE3StateSpace::StateType>()->setXYZYaw(goal_->getX(), goal_->getY(), goal_->getZ(),
      rn_distribution_(rn_generator_));
}


unsigned int MyGoalSampleableRegion::maxSampleCount(void) const {
  return maxSampleCount_;
}


void MyGoalSampleableRegion::setMaxSampleCount(unsigned int max_sample_count) {
  maxSampleCount_ = max_sample_count;
}


bool MyGoalSampleableRegion::isSatisfied(const ob::State *st) const {
  return isSatisfied(st, NULL);
}


bool MyGoalSampleableRegion::isSatisfied(const ob::State *st, double *distance) const {
  double d2g = distanceGoal(st);

  if (distance)
    *distance = d2g;

  return d2g <= threshold_;
}


void MyGoalSampleableRegion::setGoalState(const ob::ScopedState<MySE3StateSpace>& st) {
  goal_ = st;
}


const ob::State* MyGoalSampleableRegion::getGoalState() const {
  return goal_.get();
}


double MyGoalSampleableRegion::getGoalX() const {
  return goal_->getX();
}


double MyGoalSampleableRegion::getGoalY() const {
  return goal_->getY();
}


double MyGoalSampleableRegion::getGoalZ() const {
  return goal_->getZ();
}


void MyGoalSampleableRegion::printMyGoalSampleableRegion() const {
  std::cout << "Goal Sampleable Region" << std::endl;
  std::cout << "  State (in meter): " << getGoalX() << " " << getGoalY() << " "
            << getGoalZ() << std::endl;
  std::cout << "  Goal Threshold: " << threshold_ << " m" << std::endl;
}


double MyGoalSampleableRegion::distanceGoal(const ob::State *st) const {
  return si_->getStateSpace()->as<MySE3StateSpace>()->euclidean_distance(st, goal_.get());
}


} // namespace planning

} // namespace intel_wind
