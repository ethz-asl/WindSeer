/*!
 * \file params.hpp
 *
 *  Created on: Nov 20, 2018
 *      Author: Florian Achermann, ASL
 */

#ifndef INTEL_WIND__PARAMS_HPP_
#define INTEL_WIND__PARAMS_HPP_

namespace intel_wind {

namespace planning {

/** \brief Dimensions of box representing airplane (length x width x height).
 *  29.9m to avoid special cases if bounding box has the same size as the map resolution.
 */
const std::vector<double> airplane_bb_param = { 9.9, 9.9, 9.9 };

/** \brief Airspeed of the airplane */
const double v_air_param = 15;  // m/s

const double goal_threshold = 0.0; // m

const int max_num_goal_states = 1;

const double max_pitch = 0.261799; // rad

/** Cell size in meter for a projection. Used for some planners (PDST, KPIECE). */
static double proj_cellsize[3] = { 25.0 /* x */, 25.0 /* y */, 25.0 /* z */};

} // namespace planning

} // namespace intel_wind

#endif // INTEL_WIND__PARAMS_HPP_
