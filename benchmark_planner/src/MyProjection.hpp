/*!
 * \file MyProjection.hpp
 *
 *  Created on: Nov, 20 2018
 *      Author: Florian Achermann, ASL
 */

#ifndef INTEL_WIND__MY_PROJECTION_HPP_
#define INTEL_WIND__MY_PROJECTION_HPP_

#include <cstring>

#include <ompl/base/ProjectionEvaluator.h>
#include <ompl/base/State.h>
#include <ompl/base/StateSpace.h>

#include "MySE3StateSpace.hpp"

namespace intel_wind {

namespace planning {

namespace ob = ompl::base;

/** \brief MyProjection
 * A Projection to 3 dimensional euclidean space (from SE3 space, DubinsAirplane1 space, and DubinsAirplane2 space).
 */
class MyProjection : public ob::ProjectionEvaluator {
 public:
  /*! \brief Constructor */
  MyProjection(const ob::StateSpacePtr &space, double proj_cellsize_param[3])
      : ob::ProjectionEvaluator(space),
        m_proj_cellsize_ { proj_cellsize_param[0], proj_cellsize_param[1], proj_cellsize_param[2] } {
  }


  /** \brief getDimension
   * Get the dimension of the projection (3).
   */
  virtual unsigned int getDimension(void) const override {
    return dim_;
  }


  /** \brief defaultCellSizes
   * Set the default cell size of the projection.
   */
  virtual void defaultCellSizes(void) override {
    cellSizes_.resize(dim_);
    cellSizes_[0] = m_proj_cellsize_[0];
    cellSizes_[1] = m_proj_cellsize_[1];
    cellSizes_[2] = m_proj_cellsize_[2];
  }


  /** \brief project
   * Project input state to the euclidean space.
   */
  virtual void project(const ob::State *state, ob::EuclideanProjection& projection) const override {
    std::memcpy(&projection(0),
                state->as<typename MySE3StateSpace::StateType>()->getPosValuePointer(),
                dim_ * sizeof(double));
  }

 private:
  /** \brief m_proj_cellsize_
   * Default cell size of the projection.
   */
  double m_proj_cellsize_[3];

  /** \brief dim_
   * Dimension of the projected state.
   */
  const int dim_ = 3;
};

} // namespace planning

} // namespace intel_wind

#endif /* INTEL_WIND__MY_PROJECTION_HPP_ */
