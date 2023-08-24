/*!
 * 	\file MyStateValidityCheckerClass.cpp
 *
 *  Created on: Nov, 22 2018
 *      Author: Florian Achermann, ASL
 */

#include "MyStateValidityCheckerClass.hpp"

#include <chrono>
#include <iostream>

#include <boost/math/constants/constants.hpp>

#include "MySE3StateSpace.hpp"

namespace intel_wind {

namespace planning {

MyStateValidityCheckerClass::MyStateValidityCheckerClass(const ob::SpaceInformationPtr& si_b, const HeightMapClass& map)
    : ob::StateValidityChecker(si_b),
      height_map_(map){
}


MyStateValidityCheckerClass::~MyStateValidityCheckerClass() {
}


bool MyStateValidityCheckerClass::isValid(const ob::State *state) const {
  // assure state to check is within state space bounds
  if (!(si_->satisfiesBounds(state))) {
    return false;
  }

  // check all cells in the bounding box
  bool collisionDetected = height_map_.collide(state->as<typename MySE3StateSpace::StateType>()->getX(),
                                               state->as<typename MySE3StateSpace::StateType>()->getY(),
                                               state->as<typename MySE3StateSpace::StateType>()->getZ());
  return !collisionDetected;
}

} // namespace planning

} // namespace intel_wind
