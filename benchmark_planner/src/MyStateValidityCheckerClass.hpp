/*!
 * 	\file MyStateValidityCheckerClass.hpp
 *
 * One state validity checking method is based on the flexible collision library (FCL)
 * https://github.com/flexible-collision-library/fcl
 * FCL: A general purpose library for collision and proximity queries
 * Pan, Chitta and Manocha, 2012
 *
 *  Created on: Nov, 22 2018
 *      Author: Florian Achermann, ASL
 */

#ifndef INTEL_WIND__MY_STATE_VALIDITY_CHECKER_CLASS_HPP_
#define INTEL_WIND__MY_STATE_VALIDITY_CHECKER_CLASS_HPP_

#include <memory>

#include <boost/shared_ptr.hpp>

#include <ompl/base/State.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/StateValidityChecker.h>

#include "HeightMapClass.hpp"

namespace intel_wind {

namespace planning {

namespace ob = ompl::base;

/** \brief MyStateValidityCheckerClass
 * Class for state validation.
 */
class MyStateValidityCheckerClass : public ob::StateValidityChecker {
 public:
  /** \brief Constructor*/
  MyStateValidityCheckerClass(const ob::SpaceInformationPtr& si_b, const HeightMapClass& map);

  /** \brief Destructor*/
  virtual ~MyStateValidityCheckerClass() override;

  /** \brief isValid
   * Check if the input state is valid.
   */
  virtual bool isValid(const ob::State *state) const override;

 private:
  /** \brief Shared pointer of the MapManager. */
  HeightMapClass height_map_;
};

} // namespace planning

} // namespace intel_wind

#endif /* INTEL_WIND__MY_STATE_VALIDITY_CHECKER_CLASS_HPP_ */
