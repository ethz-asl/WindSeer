/*
 * HeightMapClass.cpp
 *
 *  Created on: Nov 21, 2018
 *      Author: Florian Achermann, ASL
 */

#include "HeightMapClass.hpp"

#include <cmath>
#include <cstddef>

#include "params.hpp"

namespace intel_wind {

namespace planning {

HeightMapClass::HeightMapClass() :
    max_x_(0.0),
    min_x_(0.0),
    max_y_(0.0),
    min_y_(0.0),
    max_z_(0.0),
    min_z_(0.0),
    n_x_(0),
    resolution_inverse_(0.0),
    resolution_(0.0),
    height_data_(),
    bounding_box_(airplane_bb_param),
    offset_vector_x_(),
    offset_vector_y_() {
  offset_vector_x_.push_back(0.0);
  offset_vector_y_.push_back(0.0);
}


bool HeightMapClass::setHeightMapData(const std::vector<float>& map, int n_x, double max_z, double resolution) {
  // copy values of the height map and precompute values.
  height_data_ = map;
  resolution_inverse_ = 1.0 / resolution;
  resolution_ = resolution;
  n_x_ = n_x;

  max_x_ = n_x * resolution;
  min_x_ = 0.0;
  max_y_ = map.size() / n_x * resolution;
  min_y_ = 0.0;
  min_z_ = 0.0;
  max_z_ = max_z;

  updateOffsetVectors();
  return true;
}


bool HeightMapClass::collide(double x, double y, double z) const {
  if (std::isnan(x) || std::isnan(y) || std::isnan(z))
    return true;

  const double z_check = z - bounding_box_[2] * 0.5;

  double x_check, y_check;

  // loop over all cells which are inside the bounding box
  for (auto offset_x: offset_vector_x_) {
    x_check = x + offset_x;
    for (auto offset_y: offset_vector_y_) {
      y_check = y + offset_y;

      // check if point is inside the bounding box
      if (x_check < min_x_ ||
          x_check > max_x_ ||
          y_check < min_y_ ||
          y_check > max_y_)
        return true;

      // point is inside the map so compare z to the map z value.
      const size_t idx =
          floor((x_check - min_x_) * resolution_inverse_) +
          floor((y_check - min_y_) * resolution_inverse_) * n_x_;

      if (height_data_[idx] > z_check)
        return true;
    }
  }

  // no collision detected
  return false;
}


double HeightMapClass::getTerrainHeight(double x, double y) const {
  if (std::isnan(x) || std::isnan(y))
      return 0.0;

  // point is inside the map so compare z to the map z value.
  const size_t idx =
      floor((x - min_x_) * resolution_inverse_) +
      floor((y - min_y_) * resolution_inverse_) * n_x_;

  return height_data_[idx];
}


double HeightMapClass::getMaxX() const {
  return max_x_;
}


double HeightMapClass::getMaxY() const {
  return max_y_;
}


double HeightMapClass::getMaxZ() const {
  return max_z_;
}


double HeightMapClass::getMinX() const {
  return min_x_;
}


double HeightMapClass::getMinY() const {
  return min_y_;
}


double HeightMapClass::getMinZ() const {
  return min_z_;
}


void HeightMapClass::updateOffsetVectors() {
  // check if resolution is valid
  if (resolution_ <= 0.0 || bounding_box_.empty())
    return;

  offset_vector_x_.clear();
  offset_vector_y_.clear();

  // update check vector values
  int nx = bounding_box_[0] * resolution_inverse_;
  int ny = bounding_box_[1] * resolution_inverse_;

  for (int i = 0; i <= nx; ++i) {
    offset_vector_x_.push_back(-0.5 * bounding_box_[0] + i * resolution_);
  }

  for (int i = 0; i <= ny; ++i) {
    offset_vector_y_.push_back(-0.5 * bounding_box_[1] + i * resolution_);
  }

  if (offset_vector_x_.back() != 0.5 * bounding_box_[0])
    offset_vector_x_.push_back(0.5 * bounding_box_[0]);

  if (offset_vector_y_.back() != 0.5 * bounding_box_[1])
    offset_vector_y_.push_back(0.5 * bounding_box_[1]);
}


} /* namespace planning */

} /* namespace intel_wind */
