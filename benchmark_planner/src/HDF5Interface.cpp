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
  H5Literate_by_name(file_.getLocId(), ".", H5_INDEX_NAME, H5_ITER_NATIVE, NULL, iter_func, &sample_names_, H5P_DEFAULT);
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
  out.sample_name = sample_names_[idx];

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

  // allocate buffer to read the data
  float wind[3][out.nz][out.ny][out.nx];
  float turbulence[out.nz][out.ny][out.nx];
  float terrain[out.ny][out.nx];

  // get the terrain
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

  // get the reference turbulence
  out.reference.model_name = "reference";
  dataset = group.openDataSet("reference/turbulence");
  if (dataset.getStorageSize() != (out.nx * out.ny * out.nz * 4)) {
    printf("getSample: Sample %d contains invalid data\n", idx);
    return HDF5Interface::Sample();
  }
  dataset.read(turbulence, H5::PredType::NATIVE_FLOAT);
  out.reference.turbulence.clear();
  out.reference.turbulence.reserve(out.nz * out.ny * out.nx);
  for (int i = 0; i < out.nz; i++) {
    for (int j = 0; j < out.ny; j++) {
      for (int k = 0; k < out.nx; k++) {
        out.reference.turbulence.push_back(turbulence[i][j][k]);
      }
    }
  }

  // get the reference wind
  dataset = group.openDataSet("reference/wind");
  if (dataset.getStorageSize() != (3 * out.nx * out.ny * out.nz * 4)) {
    printf("getSample: Sample %d contains invalid data\n", idx);
    return HDF5Interface::Sample();
  }
  dataset.read(wind, H5::PredType::NATIVE_FLOAT);
  out.reference.wind_x.clear();
  out.reference.wind_y.clear();
  out.reference.wind_z.clear();
  out.reference.wind_x.reserve(out.nz * out.ny * out.nx);
  out.reference.wind_y.reserve(out.nz * out.ny * out.nx);
  out.reference.wind_z.reserve(out.nz * out.ny * out.nx);
  for (int i = 0; i < out.nz; i++) {
    for (int j = 0; j < out.ny; j++) {
      for (int k = 0; k < out.nx; k++) {
          out.reference.wind_x.push_back(wind[0][i][j][k]);
          out.reference.wind_y.push_back(wind[1][i][j][k]);
          out.reference.wind_z.push_back(wind[2][i][j][k]);
      }
    }
  }

  // determin the different models used for prediction
  H5::Group predictions_group = group.openGroup("predictions");
  std::vector<std::string> model_names;
  H5Literate_by_name(predictions_group.getLocId(), ".", H5_INDEX_NAME, H5_ITER_NATIVE, NULL, iter_func, &model_names, H5P_DEFAULT);

  // loop through the different models and extract the predictions
  for(auto const& name: model_names) {
    H5::Group model_group = predictions_group.openGroup(name.c_str());
    Flow prediction;
    prediction.model_name = name;

    // get the predicted wind
    dataset = model_group.openDataSet("wind");
    if (dataset.getStorageSize() != (3 * out.nx * out.ny * out.nz * 4)) {
      printf("getSample: Sample %d contains invalid data\n", idx);
      return HDF5Interface::Sample();
    }
    dataset.read(wind, H5::PredType::NATIVE_FLOAT);
    prediction.wind_x.clear();
    prediction.wind_y.clear();
    prediction.wind_z.clear();
    prediction.wind_x.reserve(out.nz * out.ny * out.nx);
    prediction.wind_y.reserve(out.nz * out.ny * out.nx);
    prediction.wind_z.reserve(out.nz * out.ny * out.nx);
    for (int i = 0; i < out.nz; i++) {
      for (int j = 0; j < out.ny; j++) {
        for (int k = 0; k < out.nx; k++) {
            prediction.wind_x.push_back(wind[0][i][j][k]);
            prediction.wind_y.push_back(wind[1][i][j][k]);
            prediction.wind_z.push_back(wind[2][i][j][k]);
        }
      }
    }

    // get the predicted turbulence
    dataset = model_group.openDataSet("turbulence");
    if (dataset.getStorageSize() != (out.nx * out.ny * out.nz * 4)) {
      printf("getSample: Sample %d contains invalid data\n", idx);
      return HDF5Interface::Sample();
    }
    dataset.read(turbulence, H5::PredType::NATIVE_FLOAT);
    prediction.turbulence.clear();
    prediction.turbulence.reserve(out.nz * out.ny * out.nx);
    for (int i = 0; i < out.nz; i++) {
      for (int j = 0; j < out.ny; j++) {
        for (int k = 0; k < out.nx; k++) {
            prediction.turbulence.push_back(turbulence[i][j][k]);
        }
      }
    }

    out.predictions.push_back(prediction);
  }

  return out;
}

herr_t iter_func(hid_t loc_id, const char *name, const H5L_info_t *info,
                 void *operator_data) {
  H5O_info_t infobuf;
  H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT);
  std::string str(name);
  std::vector<std::string> *od = (std::vector<std::string> *)operator_data;
  od->push_back(str);

  return 0;
}

} /* namespace planning */

} /* namespace intwl_wind */
