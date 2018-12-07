/*
 * HDF5Interface.hpp
 *
 *  Created on: Dec 4, 2018
 *      Author: Florian Achermann, ASL
 */

#ifndef INTEL_PLANNING__HDF5_INTERFACE_HPP_
#define INTEL_PLANNING__HDF5_INTERFACE_HPP_

#include <string>
#include <vector>

#include <H5Cpp.h>

namespace intel_wind {

namespace planning {

/** \brief HDF5Interface
 */
class HDF5Interface {
public:
  struct Sample {
    int nx = 0;
    int ny = 0;
    int nz = 0;

    float resolution_horizontal = 0.0f;
    float resolution_vertical = 0.0f;

    std::vector<float> terrain = {{}};

    std::vector<float> turbulence = {{}};
    std::vector<float> turbulence_reference = {{}};

    std::vector<float> wind_x = {{}};
    std::vector<float> wind_y = {{}};
    std::vector<float> wind_z = {{}};

    std::vector<float> wind_x_reference = {{}};
    std::vector<float> wind_y_reference = {{}};
    std::vector<float> wind_z_reference = {{}};

  };
  /** \brief Constructor */
  HDF5Interface();

  ~HDF5Interface();

  void init(std::string filename);

  int getNumberSamples() const;

  std::vector<std::string>& getSampleNames();

  Sample getSample(int idx) const;
private:
  H5::H5File file_;

  std::vector<std::string> sample_names_;

};

herr_t iter_func(hid_t loc_id, const char *name, const H5L_info_t *info,
                 void *operator_data);

} /* namespace planning */

} /* namespace intel_wind */

#endif /* INTEL_PLANNING__HDF5_INTERFACE_HPP_ */
