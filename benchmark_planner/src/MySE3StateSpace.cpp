/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2010, Rice University
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Rice University nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/*
 * MySE3StateSpace.cpp
 *
 * Based on the code written by OMPL/RiceUniversity but modified by the
 * ETHZ/ASL.
 *
 *  Created on: Nov 7, 2016
 *      Author: Florian Achermann, ASL
 *
 *      Note: See header file for more information on definitions and states.
 */


#include "MySE3StateSpace.hpp"

#include <iostream>
#include <cstring>

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/ProjectionEvaluator.h>
#include <ompl/tools/config/MagicConstants.h>

namespace intel_wind {

namespace planning {

/*-----------------------------------------------------------------------------------------------*/
/*- StateType -----------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------*/

MySE3StateSpace::StateType::StateType() : ob::CompoundStateSpace::StateType(),
                                          roll_(0.0),
                                          pitch_(0.0),
                                          yaw_(0.0) {
}


double MySE3StateSpace::StateType::getX() const {
  return as<ob::RealVectorStateSpace::StateType>(0)->values[0];
}


double MySE3StateSpace::StateType::getY() const {
  return as<ob::RealVectorStateSpace::StateType>(0)->values[1];
}


double MySE3StateSpace::StateType::getZ() const {
  return as<ob::RealVectorStateSpace::StateType>(0)->values[2];
}


double MySE3StateSpace::StateType::getYaw() const {
  // copy the value of the state into Eigen::Quaternion
  Eigen::Quaternionf q(
          as<ob::SO3StateSpace::StateType>(1)->w,
          as<ob::SO3StateSpace::StateType>(1)->x,
          as<ob::SO3StateSpace::StateType>(1)->y,
          as<ob::SO3StateSpace::StateType>(1)->z);

  // compute the yaw angle and return it
  auto euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
  roll_ = euler[0];
  pitch_ = euler[1];
  roll_ = euler[2];

  return yaw_;
}


const ob::SO3StateSpace::StateType& MySE3StateSpace::StateType::rotation() const {
  return *as<ob::SO3StateSpace::StateType>(1);
}


ob::SO3StateSpace::StateType& MySE3StateSpace::StateType::rotation() {
  return *as<ob::SO3StateSpace::StateType>(1);
}


void MySE3StateSpace::StateType::setX(double x) {
  as<ob::RealVectorStateSpace::StateType>(0)->values[0] = x;
}


void MySE3StateSpace::StateType::setY(double y) {
  as<ob::RealVectorStateSpace::StateType>(0)->values[1] = y;
}


void MySE3StateSpace::StateType::setZ(double z) {
  as<ob::RealVectorStateSpace::StateType>(0)->values[2] = z;
}


void MySE3StateSpace::StateType::setYaw(double yaw) {
  Eigen::Quaternionf q = Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitY())
        * Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
  as<ob::SO3StateSpace::StateType>(1)->x = q.x();
  as<ob::SO3StateSpace::StateType>(1)->y = q.y();
  as<ob::SO3StateSpace::StateType>(1)->z = q.z();
  as<ob::SO3StateSpace::StateType>(1)->w = q.w();
}


void MySE3StateSpace::StateType::setXYZ(double x, double y, double z) {
  setX(x);
  setY(y);
  setZ(z);
}


void MySE3StateSpace::StateType::setXYZYaw(double x, double y, double z, double yaw) {
  setX(x);
  setY(y);
  setZ(z);
  setYaw(yaw);
}


const double* MySE3StateSpace::StateType::getPosValuePointer() const {
  return as<ob::RealVectorStateSpace::StateType>(0)->values;
}


void MySE3StateSpace::StateType::printState(const std::string& msg) const {
  std::cout << msg << "state (x,y,z,yaw): " << this->getX() << " " << this->getY() << " " << this->getZ() << " "
            << this->getYaw() << std::endl << std::endl;
}


/*-----------------------------------------------------------------------------------------------*/
/*- MySE3StateSpace -----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------*/

MySE3StateSpace::MySE3StateSpace() : CompoundStateSpace() {
  setName("SE3" + getName());
  type_ = ob::STATE_SPACE_SE3;
  addSubspace(ob::StateSpacePtr(new ob::RealVectorStateSpace(3)), 1.0);
  addSubspace(ob::StateSpacePtr(new ob::SO3StateSpace()), 1.0);
  lock();
}


void MySE3StateSpace::setBounds(const ob::RealVectorBounds & bounds) {
  as<ob::RealVectorStateSpace>(0)->setBounds(bounds);
}


const ob::RealVectorBounds& MySE3StateSpace::getBounds() const {
  return as<ob::RealVectorStateSpace>(0)->getBounds();
}


ob::State* MySE3StateSpace::allocState() const {
    StateType *state = new StateType();
    allocStateComponents(state);
    return state;
}


void MySE3StateSpace::freeState(ob::State *state) const {
    CompoundStateSpace::freeState(state);
}


void MySE3StateSpace::registerProjections()
{
  class SE3DefaultProjection : public ob::ProjectionEvaluator
  {
  public:
    SE3DefaultProjection(const StateSpace *space) : ProjectionEvaluator(space) {
    }

    virtual unsigned int getDimension() const {
        return 3;
    }

    virtual void defaultCellSizes() {
        cellSizes_.resize(3);
        bounds_ = space_->as<MySE3StateSpace>()->getBounds();
        cellSizes_[0] = (bounds_.high[0] - bounds_.low[0]) / ompl::magic::PROJECTION_DIMENSION_SPLITS;
        cellSizes_[1] = (bounds_.high[1] - bounds_.low[1]) / ompl::magic::PROJECTION_DIMENSION_SPLITS;
        cellSizes_[2] = (bounds_.high[2] - bounds_.low[2]) / ompl::magic::PROJECTION_DIMENSION_SPLITS;
    }

    virtual void project(const ob::State *state, ob::EuclideanProjection &projection) const {
        memcpy(&projection(0), state->as<MySE3StateSpace::StateType>()->as<ob::RealVectorStateSpace::StateType>(0)->values, 3 * sizeof(double));
    }
  };
  registerDefaultProjection(ob::ProjectionEvaluatorPtr(dynamic_cast<ob::ProjectionEvaluator*>(new SE3DefaultProjection(this))));
}


double MySE3StateSpace::distance(const ob::State* state1, const ob::State* state2) const {
  return euclidean_distance(state1, state2);
}


double MySE3StateSpace::getEuclideanExtent() const {
  /* For the DubinsAirplane2StateSpace this computes:
   * R3_max_extent = sqrt(bound_x^2 + bound_y^2 + bound_z^2) */
  return components_[0]->getMaximumExtent();
}


double MySE3StateSpace::euclidean_distance(const ob::State* state1, const ob::State* state2) const {
  const MySE3StateSpace::StateType *mySE3state1 = state1->as<MySE3StateSpace::StateType>();
  const MySE3StateSpace::StateType *mySE3state2 = state2->as<MySE3StateSpace::StateType>();

  const double eucl_dist =
      ((mySE3state1->getX() - mySE3state2->getX()) * (mySE3state1->getX() - mySE3state2->getX())) +
      ((mySE3state1->getY() - mySE3state2->getY()) * (mySE3state1->getY() - mySE3state2->getY())) +
      ((mySE3state1->getZ() - mySE3state2->getZ()) * (mySE3state1->getZ() - mySE3state2->getZ()));

  return sqrt(eucl_dist);
}


void MySE3StateSpace::printStateSpaceProperties() const {
  std::cout << "MySE3StateSpace" << std::endl;
  std::cout << "  State space bounds (in meter and radian): [" << this->getBounds().low[0] << " " << this->getBounds().high[0]
            << "], [" << this->getBounds().low[1] << " " << this->getBounds().high[1] << "], [" << this->getBounds().low[2]
            << " " << this->getBounds().high[2] << "], [" << "-pi pi" << ")" << std::endl;
  std::cout << "  Use wind: " << this->_use_wind;
  std::cout << std::endl << std::endl;
}


const bool MySE3StateSpace::getUseWind() const {
	return _use_wind;
}


void MySE3StateSpace::setUseWind(const bool use_wind) {
	_use_wind = use_wind;
}

} // namespace planning

} // namespace intel_wind
