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

/*!
 * \file MySE3StateSpace.h
 *
 * Custom implementation of the SE3 state space.
 * The purpose is that the state has the same functions
 * as the DA1 and DA2 state space.
 *
 * Based on the code written by OMPL/RiceUniversity but modified by the
 * ETHZ/ASL.
 *
 *  Created on: Nov 20, 2018
 *      Author: Florian Achermann, ASL
 */

#ifndef INTEL_WIND__MYSE3_STATE_SPACE_
#define INTEL_WIND__MYSE3_STATE_SPACE_

#include <string>

#include <Eigen/Geometry>

#include <ompl/base/State.h>
#include <ompl/base/StateSpace.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/base/spaces/SO3StateSpace.h>

namespace ob = ompl::base;

namespace intel_wind {

namespace planning {

/** \brief MySE3StateSpace
 * A state space representing SE(3), basically a copy from the SE3StateSpace from ompl
 * with additional functions of the StateType to allow templating the state.
 */
class MySE3StateSpace : public ob::CompoundStateSpace
{
public:
  /** \brief
   * A state in SE(3): position = (x, y, z), quaternion = (x, y, z, w)
   * The yaw angle of the orientation can be directly accessed and set.
   */
  class StateType : public ob::CompoundStateSpace::StateType
  {
  public:
    /** \brief Constructor */
    StateType();

    /** \brief getX
     * Get the X component of the state
     */
    double getX() const;

    /** \brief getY
     * Get the Y component of the state
     */
    double getY() const;

    /** \brief getZ
     * Get the Z component of the state
     */
    double getZ() const;

    /** \brief getYaw
     * Get the heading/yaw component of the state
     */
    double getYaw() const;

    /** \brief rotation
     * Get the rotation component of the state
     */
    const ob::SO3StateSpace::StateType& rotation() const;

    /** \brief rotation
     * Get the rotation component of the state and allow changing it as well
     */
    ob::SO3StateSpace::StateType& rotation();

    /** \brief setX
     * Set the X component of the state
     */
    void setX(double x);

    /** \brief setY
     * Set the Y component of the state
     */
    void setY(double y);

    /** \brief setZ
     * Set the Z component of the state
     */
    void setZ(double z);

    /** \brief setYaw
     * Set the Z component of the state
     */
    void setYaw(double yaw);

    /** \brief setXYZ
     * Set the X, Y and Z components of the state
     */
    void setXYZ(double x, double y, double z);

    /** \brief setXYZYaw
     * Set the X, Y, Z and Yaw components of the state
     */
    void setXYZYaw(double x, double y, double z, double yaw);

    /** \brief getPosValuePointer
     * Get a pointer to the position values.
     */
    const double* getPosValuePointer() const;

    /** \brief printState
     * Print the state together with a message.
     */
    void printState(const std::string& msg = "") const;

  protected:
    /** roll_
     * Variable to cache the value of a roll angle.
     */
    mutable double roll_;

    /** pitch_
     * Variable to cache the value of a pitch angle.
     */
    mutable double pitch_;

    /** yaw_
     * Variable to cache the value of a yaw angle.
     */
    mutable double yaw_;
  };

  /** \brief Constructor */
  MySE3StateSpace();

  /** \brief Destructor */
  virtual ~MySE3StateSpace() {}

  /** \brief setBounds
   * Set the bounds of this state space.
   */
  void setBounds(const ob::RealVectorBounds & bounds);

  /** \brief getBounds
   * Get the bounds of this state space.
   */
  const ob::RealVectorBounds& getBounds() const;

  /** \brief allocState
   * Allocate a state pointer.
   */
  virtual ob::State* allocState() const;

  /** \brief freeState
   * Free the input state pointer.
   */
  virtual void freeState(ob::State *state) const;

  /** \brief registerProjections
   * Register the default projection for this state space.
   */
  virtual void registerProjections();

  /** \brief distance
   * Returns the length of straight line path connecting \a state1 and \a state2.
   */
  virtual double distance(const ob::State* state1, const ob::State* state2) const override;

  /** \brief euclidean_distance
   * Returns euclidean distance between \a state1 and \a state2
   */
  double euclidean_distance(const ob::State* state1, const ob::State* state2) const;

  /** \brief getEuclideanExtent
   * Get the maximum extent of the RealVectorStateSpace part of the SE3StateSpace. */
  double getEuclideanExtent() const;

  /** \brief printStateSpaceProperties
   * Print the properties of the state space.
   */
  void printStateSpaceProperties() const;

  /** \brief getUseWind
   * Return if the wind is used.
   */
  const bool getUseWind() const;

  /** \brief setUseWind
   * Set if the wind should be used for the path computation.
   */
  void setUseWind(const bool use_wind);

private:
  bool _use_wind = false;
};

} // namespace planning

} // namespace intel_wind

#endif /* INTEL_WIND__MYSE3_STATE_SPACE_ */
