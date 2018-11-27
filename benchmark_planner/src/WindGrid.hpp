/*
 * MeteoGridClass.hpp
 *
 *  Created on: Nov 22, 2018
 *      Author: Florian Achermann, ASL
 */

#ifndef INTEL_PLANNING__WIND_GRID_HPP_
#define INTEL_PLANNING__WIND_GRID_HPP_

#include<vector>

namespace intel_wind {

namespace planning {

struct WindGridGeometry {
  int nx = 0;
  int ny = 0;
  int nz = 0;
  double resolution_hor = 0.0;
  double resolution_ver = 0.0;
  double min_x = 0.0;
  double min_y = 0.0;
  double min_z = 0.0;
};

/** \brief WindGrid
 */
class WindGrid {
public:
  /** \brief Constructor */
    WindGrid();

  /** \brief getMeteoData
   * Get the data of the grid point at (x,y,z). Currently no interpolation is done, the data from the
   * nearest grid point is returned.
   *
   * @param[in] x: Input x value [m]
   * @param[in] y: Input y value [m]
   * @param[in] z: Input z value [m]
   * @param[out] u: The wind in x-direction [m/s]
   * @param[out] v: The wind in y-direction [m/s]
   * @param[out] w: The wind in z-direction [m/s]
   * @return: Indicates if successfully data is extracted.
   */
  bool getWind(double x, double y, double z, float* u, float* v, float* w) const;

  /** \brief getMeteoData
   * Insert a meteo grid message and precompute values.
   */
  bool setWindGrid(const std::vector<float>& u, const std::vector<float>& v, const std::vector<float>& w, const WindGridGeometry& geo);

  /** \brief getMaxWindMagnitudeX
   * Get the value of the maximum magnitude of the wind in x-direction of the meteo grid.
   */
  double getMaxWindMagnitudeX() const;

  /** \brief getMaxWindMagnitudeY
   * Get the value of the maximum magnitude of the wind in y-direction of the meteo grid.
   */
  double getMaxWindMagnitudeY() const;

  /** \brief getMaxWindMagnitudeZ
   * Get the value of the maximum magnitude of the wind in z-direction of the meteo grid.
   */
  double getMaxWindMagnitudeZ() const;

  /** \brief getResolution
   * Get the resolution of the meteo grid in x/y direction.
   */
  double getResolution() const;

  /** \brief isSet
   * Returns if the class actually contains some meaningful data.
   * Checks that n_x, n_y, n_z is nonzero.
   */
  bool isSet() const;

private:
  /** \brief Extracted minimum x value of the meteo grid. */
  double min_x_ = 0.0;

  /** \brief Extracted maximum x value of the meteo grid. */
  double max_x_ = 0.0;

  /** \brief Extracted minimum y value of the meteo grid. */
  double min_y_ = 0.0;

  /** \brief Extracted maximum y value of the meteo grid. */
  double max_y_ = 0.0;

  /** \brief Extracted minimum z value of the meteo grid. */
  double min_z_ = 0.0;

  /** \brief Extracted maximum z value of the meteo grid. */
  double max_z_ = 0.0;

  /** \brief Resolution of the meteo grid in x/y direction. */
  double resolution_hor_ = 0.0;

  /** \brief Precomputed value of the inverse of the resolution of the meteo grid. */
  double resolution_inverse_hor_ = 0.0;

  /** \brief Resolution of the meteo grid in z direction. */
  double resolution_ver_ = 0.0;

  /** \brief Precomputed value of the inverse of the resolution of the meteo grid. */
  double resolution_inverse_ver_ = 0.0;

  /** \brief Extracted number of grid points in x direction. */
  int n_x_ = 0;

  /** \brief Extracted number of grid points in y direction. */
  int n_y_ = 0;

  /** \brief Extracted number of grid points in z direction. */
  int n_z_ = 0;

  /** \brief wind in x-direction. */
  std::vector<float> wind_x_ = {};

  /** \brief wind in y-direction. */
  std::vector<float> wind_y_ = {};

  /** \brief wind in z-direction. */
  std::vector<float> wind_z_ = {};

  /** \brief Maximum magnitude of the wind in x-direction of the meteo grid. */
  double max_wind_magnitude_x_ = 0.0;

  /** \brief Maximum magnitude of the wind in y-direction of the meteo grid. */
  double max_wind_magnitude_y_ = 0.0;

  /** \brief Maximum magnitude of the wind in z-direction of the meteo grid. */
  double max_wind_magnitude_z_ = 0.0;

  // variables used to store intermediate results in getMeteoData
  mutable int idx_3D_ = 0;
};

} /* namespace planning */

} /* namespace intel_wind */

#endif /* INTEL_PLANNING__WIND_GRID_HPP_ */
