/*
 * WindGrid.cpp
 *
 *  Created on: Nov 22, 2018
 *      Author: Florian Achermann, ASL
 */

#include "WindGrid.hpp"

#include <cmath>

namespace intel_wind {

namespace planning {

WindGrid::WindGrid() {
}


bool WindGrid::getWind(double x, double y, double z, float& u, float& v, float& w) const {
  // assign output with 0.0
  u = 0.0f;
  v = 0.0f;
  w = 0.0f;

  // check if point is in boundaries
  if ((x < min_x_) || (y < min_y_) || (z < min_z_) || (x > max_x_) || (y > max_y_) || (z > max_z_) ||
      std::isnan(x) || std::isnan(y) || std::isnan(z))
    return false;

  idx_3D_ = int((x - min_x_) * resolution_inverse_hor_) +
            int((y - min_y_) * resolution_inverse_hor_) * n_x_ +
            int((z - min_z_) * resolution_inverse_ver_) * n_x_ * n_y_;

  u = wind_x_[idx_3D_];
  v = wind_y_[idx_3D_];
  w = wind_z_[idx_3D_];

  return true;
}


bool WindGrid::setWindGrid(std::vector<float>& u, std::vector<float>& v, std::vector<float>& w, WindGridGeometry& geo) {
  // if the input is consistent
  if ((u.size() != v.size()) ||
      (u.size() != w.size()) ||
      (u.size() != geo.nx * geo.ny * geo.nz)) {
    return false;
  }

  // set the geometric properties of the grid
  n_x_ = geo.nx;
  n_y_ = geo.ny;
  n_z_ = geo.nz;

  resolution_hor_ = geo.resolution_hor;
  resolution_inverse_hor_ = 1.0 / geo.resolution_hor;
  resolution_ver_ = geo.resolution_ver;
  resolution_inverse_ver_ = 1.0 / geo.resolution_ver;

  min_x_ = geo.min_x;
  min_y_ = geo.min_y;
  min_z_ = geo.min_z;

  max_x_ = n_x_ * resolution_hor_;
  max_y_ = n_y_ * resolution_hor_;
  max_z_ = n_z_ * resolution_ver_;

  // set the data of the wind grid
  wind_x_ = u;
  wind_y_ = v;
  wind_z_ = w;

  // determine the maximum wind speeds
  for (unsigned int i = 0; i < wind_x_.size(); ++i) {
    if (fabs(wind_x_[i]) > max_wind_magnitude_x_)
      max_wind_magnitude_x_ = fabs(wind_x_[i]);
    if (fabs(wind_y_[i]) > max_wind_magnitude_y_)
      max_wind_magnitude_y_ = fabs(wind_y_[i]);
    if (fabs(wind_z_[i]) > max_wind_magnitude_z_)
      max_wind_magnitude_z_ = fabs(wind_z_[i]);
  }

  return true;
}


double WindGrid::getMaxWindMagnitudeX() const {
  return max_wind_magnitude_x_;
}


double WindGrid::getMaxWindMagnitudeY() const {
  return max_wind_magnitude_y_;
}


double WindGrid::getMaxWindMagnitudeZ() const {
  return max_wind_magnitude_z_;
}


double WindGrid::getResolution() const {
  return std::min(resolution_hor_, resolution_ver_);
}


bool WindGrid::isSet() const {
  return (n_x_ != 0) && (n_y_ != 0) && (n_z_ != 0);
}


} /* namespace planning */

} /* namespace intwl_wind */
