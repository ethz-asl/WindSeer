/*
 * HDF5Interface.cpp
 *
 *  Created on: Dec 4, 2018
 *      Author: Florian Achermann, ASL
 */

#include "HDF5Interface.hpp"

#include <cmath>
#include <stdexcept>

namespace intel_wind {

namespace planning {

HDF5Interface::HDF5Interface()
    : file_(),
      sample_names_() {
}


HDF5Interface::~HDF5Interface() {
  file_.close();
}


void HDF5Interface::init(std::string filename) {
  // open the file
  file_.openFile(filename, H5F_ACC_RDONLY);

  // get all the sample group names
  H5Literate_by_name(file_.getLocId(), ".", H5_INDEX_NAME, H5_ITER_NATIVE, NULL, iter_func, this, H5P_DEFAULT);
}


int HDF5Interface::getNumberSamples() const {
  return sample_names_.size();
}


std::vector<std::string>& HDF5Interface::getSampleNames() {
  return sample_names_;
}


HDF5Interface::Sample HDF5Interface::getSample(int idx) const {
  if (idx < 0 || idx >= getNumberSamples()) {
    throw std::invalid_argument("getSample: Invalid index");
  }

  HDF5Interface::Sample out;

  // open the group and copy the data to the output struct
  H5::Group group = file_.openGroup(sample_names_[idx].c_str());

  // get the grid information
  H5::DataSet dataset = group.openDataSet("grid_info/nx");
  dataset.read(&out.nx, H5::PredType::NATIVE_INT);

  dataset = group.openDataSet("grid_info/ny");
  dataset.read(&out.ny, H5::PredType::NATIVE_INT);

  dataset = group.openDataSet("grid_info/nz");
  dataset.read(&out.nz, H5::PredType::NATIVE_INT);

  dataset = group.openDataSet("grid_info/resolution_horizontal");
  dataset.read(&out.resolution_horizontal, H5::PredType::NATIVE_FLOAT);

  dataset = group.openDataSet("grid_info/resolution_vertical");
  dataset.read(&out.resolution_vertical, H5::PredType::NATIVE_FLOAT);

  // get the terrain
  float terrain[out.ny][out.nx];
  dataset = group.openDataSet("terrain");
  if (dataset.getStorageSize() != (out.nx * out.ny * 4)) {
    printf("getSample: Sample %d contains invalid data\n", idx);
    return HDF5Interface::Sample();
  }
  dataset.read(terrain, H5::PredType::NATIVE_FLOAT);
  out.terrain.clear();
  out.terrain.reserve(out.ny * out.nx);
  for (int i = 0; i < out.ny; i++) {
    for (int j = 0; j < out.nx; j++) {
      out.terrain.push_back(terrain[i][j]);
    }
  }

  // get the predicted turbulence
  float turbulence[out.nz][out.ny][out.nx];
  dataset = group.openDataSet("turbulence");
  if (dataset.getStorageSize() != (out.nx * out.ny * out.nz * 4)) {
    printf("getSample: Sample %d contains invalid data\n", idx);
    return HDF5Interface::Sample();
  }
  dataset.read(turbulence, H5::PredType::NATIVE_FLOAT);
  out.turbulence.clear();
  out.turbulence.reserve(out.nz * out.ny * out.nx);
  for (int i = 0; i < out.nz; i++) {
    for (int j = 0; j < out.ny; j++) {
      for (int k = 0; k < out.nx; k++) {
        out.turbulence.push_back(turbulence[i][j][k]);
      }
    }
  }

  // get the reference turbulence
  dataset = group.openDataSet("turbulence_reference");
  if (dataset.getStorageSize() != (out.nx * out.ny * out.nz * 4)) {
    printf("getSample: Sample %d contains invalid data\n", idx);
    return HDF5Interface::Sample();
  }
  dataset.read(turbulence, H5::PredType::NATIVE_FLOAT);
  out.turbulence_reference.clear();
  out.turbulence_reference.reserve(out.nz * out.ny * out.nx);
  for (int i = 0; i < out.nz; i++) {
    for (int j = 0; j < out.ny; j++) {
      for (int k = 0; k < out.nx; k++) {
        out.turbulence_reference.push_back(turbulence[i][j][k]);
      }
    }
  }

  // get the predicted wind
  float wind[3][out.nz][out.ny][out.nx];
  dataset = group.openDataSet("wind");
  if (dataset.getStorageSize() != (3 * out.nx * out.ny * out.nz * 4)) {
    printf("getSample: Sample %d contains invalid data\n", idx);
    return HDF5Interface::Sample();
  }
  dataset.read(wind, H5::PredType::NATIVE_FLOAT);
  out.wind_x.clear();
  out.wind_y.clear();
  out.wind_z.clear();
  out.wind_x.reserve(out.nz * out.ny * out.nx);
  out.wind_y.reserve(out.nz * out.ny * out.nx);
  out.wind_z.reserve(out.nz * out.ny * out.nx);
  for (int i = 0; i < out.nz; i++) {
    for (int j = 0; j < out.ny; j++) {
      for (int k = 0; k < out.nx; k++) {
          out.wind_x.push_back(wind[0][i][j][k]);
          out.wind_y.push_back(wind[1][i][j][k]);
          out.wind_z.push_back(wind[2][i][j][k]);
      }
    }
  }

  // get the reference wind
  dataset = group.openDataSet("wind_reference");
  if (dataset.getStorageSize() != (3 * out.nx * out.ny * out.nz * 4)) {
    printf("getSample: Sample %d contains invalid data\n", idx);
    return HDF5Interface::Sample();
  }
  dataset.read(wind, H5::PredType::NATIVE_FLOAT);
  out.wind_x_reference.clear();
  out.wind_y_reference.clear();
  out.wind_z_reference.clear();
  out.wind_x_reference.reserve(out.nz * out.ny * out.nx);
  out.wind_y_reference.reserve(out.nz * out.ny * out.nx);
  out.wind_z_reference.reserve(out.nz * out.ny * out.nx);
  for (int i = 0; i < out.nz; i++) {
    for (int j = 0; j < out.ny; j++) {
      for (int k = 0; k < out.nx; k++) {
          out.wind_x_reference.push_back(wind[0][i][j][k]);
          out.wind_y_reference.push_back(wind[1][i][j][k]);
          out.wind_z_reference.push_back(wind[2][i][j][k]);
      }
    }
  }

//  printf("wind_x[40][40][40]:           %.2f\n", out.wind_x[40+40*out.nx+40*out.nx*out.ny]);
//  printf("wind_x_reference[40][40][40]: %.2f\n", out.wind_x_reference[40+40*out.nx+40*out.nx*out.ny]);
//  printf("wind_y[40][40][40]:           %.2f\n", out.wind_y[40+40*out.nx+40*out.nx*out.ny]);
//  printf("wind_y_reference[40][40][40]: %.2f\n", out.wind_y_reference[40+40*out.nx+40*out.nx*out.ny]);
//  printf("wind_z[40][40][40]:           %.2f\n", out.wind_z[40+40*out.nx+40*out.nx*out.ny]);
//  printf("wind_z_reference[40][40][40]: %.2f\n", out.wind_z_reference[40+40*out.nx+40*out.nx*out.ny]);

  return out;
}


herr_t iter_func (hid_t loc_id, const char *name, const H5L_info_t *info,
                  void *operator_data) {
  H5O_info_t infobuf;
  H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT);
  std::string str(name);
  HDF5Interface *od = (HDF5Interface *)operator_data;
  od->getSampleNames().push_back(str);

  return 0;
}

} /* namespace planning */

} /* namespace intwl_wind */
