/*
 * HeightMapClass.hpp
 *
 *  Created on: Nov 21, 2018
 *      Author: Florian Achermann, ASL
 */

#ifndef INTEL_WIND__HEIGHT_MAP_CLASS_HPP_
#define INTEL_WIND__HEIGHT_MAP_CLASS_HPP_

#include <vector>

namespace intel_wind {

namespace planning {


class HeightMapClass {
public:
  /** \brief Constructor */
  HeightMapClass();

  /** \brief setHeightMapData
   * Set the map.
   */
  bool setHeightMapData(const std::vector<float>& map, int n_x, double max_z, double resolution);

  /** \brief collide
   * Check if the input point yields in a collision with the map.
   */
  bool collide(double x, double y, double z) const;

  /** \brief getMaxX
   * The the upper boundary in x direction in the global frame.
   */
  double getMaxX() const;

  /** \brief getMaxY
   * The the upper boundary in y direction in the global frame.
   */
  double getMaxY() const;

  /** \brief getMaxZ
   * The the upper boundary in z direction in the global frame.
   */
  double getMaxZ() const;

  /** \brief getMinX
   * The the lower boundary in x direction in the global frame.
   */
  double getMinX() const;

  /** \brief getMinY
   * The the lower boundary in y direction in the global frame.
   */
  double getMinY() const;

  /** \brief getMinZ
   * The the lower boundary in z direction in the global frame.
   */
  double getMinZ() const;

private:
  /** \brief updateOffsetVectors
   * Update the values of the offset vectors according to the map
   * resolution and the bounding box.
   */
  void updateOffsetVectors();

  /** \brief Upper boundary in x direction in the global frame.*/
  double max_x_;

  /** \brief Lower boundary in x direction in the global frame.*/
  double min_x_;

  /** \brief Upper boundary in y direction in the global frame.*/
  double max_y_;

  /** \brief Lower boundary in y direction in the global frame.*/
  double min_y_;

  /** \brief Upper boundary in z direction in the global frame.*/
  double max_z_;

  /** \brief Lower boundary in z direction in the global frame.*/
  double min_z_;

  /** \brief Number of cells in x direction.*/
  int n_x_;

  /** \brief Inverse of the resolution.*/
  double resolution_inverse_;

  /** \brief The resolution.*/
  double resolution_;

  /** \brief Height data in the global frame.*/
  std::vector<float> height_data_;

  /** \brief Bounding box of the airplane.*/
  std::vector<double> bounding_box_;

  /** \brief Vector including the offsets in x direction for collision checking.*/
  std::vector<double> offset_vector_x_;

  /** \brief Vector including the offsets in y direction for collision checking.*/
  std::vector<double> offset_vector_y_;
};

} /* namespace planning */

} /* namespace intel_wind */

#endif /* INTEL_WIND__HEIGHT_MAP_CLASS_HPP_ */
