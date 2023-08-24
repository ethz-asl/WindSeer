/*
 * MyNearestNeighbors.hpp
 *
 *  Created on: Nov 21, 2018
 *      Author: Florian Achermann, ASL
 */

#ifndef INTEL_WIND__MY_NEAREST_NEIGHBORS_HPP_
#define INTEL_WIND__MY_NEAREST_NEIGHBORS_HPP_

#include <cmath>
#include <limits>

#include <ompl/datastructures/NearestNeighborsLinear.h>
#include "ompl/util/Exception.h"

namespace intel_wind {

namespace planning {

template<typename _T>
class MyNearestNeighbors : public ompl::NearestNeighborsLinear<_T> {
public:
    MyNearestNeighbors() : ompl::NearestNeighborsLinear<_T>() {
  }


  virtual ~MyNearestNeighbors() {
  }

  /** \brief Set the distance function to compute the euclidean distance */
  virtual void setEuclDistanceFunction(const typename ompl::NearestNeighbors<_T>::DistanceFunction &distFun) {
    euclDistFun_ = distFun;
  }


  /** \brief Returns the 1-nearest motion from the neighborhood to the input state. */
  virtual _T nearest(const _T &data) const {
    const std::size_t sz = ompl::NearestNeighborsLinear<_T>::data_.size();
    std::size_t pos = sz;
    double dmin = 0.0;
    for (std::size_t i = 0 ; i < sz ; ++i) {
      double distance = euclDistFun_(ompl::NearestNeighborsLinear<_T>::data_[i], data);
      if (pos == sz || dmin > distance) {
        pos = i;
        dmin = distance;
      }
    }
    if (pos != sz)
      return ompl::NearestNeighborsLinear<_T>::data_[pos];
    throw ompl::Exception("No elements found in nearest neighbors data structure");
  }


  /** \brief Returns the k-nearest motions to the the input state from the motion tree. */
  virtual void nearestK(const _T &data, std::size_t k, std::vector<_T> &nbh) const {
    nbh = ompl::NearestNeighborsLinear<_T>::data_;
    if (nbh.size() > k) {

      std::partial_sort(nbh.begin(), nbh.begin() + k, nbh.end(), ElemSortTo(data, euclDistFun_));
      nbh.resize(k);
    } else {
      std::sort(nbh.begin(), nbh.end(), ElemSortTo(data, euclDistFun_));
    }
  }


  /** \brief Returns the k-nearest motions from the the input state to the motion tree. */
  virtual void nearestKFrom(const _T &data, std::size_t k, std::vector<_T> &nbh) const {
    nbh = ompl::NearestNeighborsLinear<_T>::data_;
    if (nbh.size() > k) {

      std::partial_sort(nbh.begin(), nbh.begin() + k, nbh.end(), ElemSortFrom(data, euclDistFun_));
      nbh.resize(k);
    } else {
      std::sort(nbh.begin(), nbh.end(), ElemSortFrom(data, euclDistFun_));
    }
  }


private:
  struct ElemSortFrom {
    ElemSortFrom(const _T &e, const typename ompl::NearestNeighbors<_T>::DistanceFunction &df) : e_(e), df_(df) {}

    bool operator()(const _T &a, const _T &b) const {
        return df_(e_, a) < df_(e_, b);
    }

    const _T &e_;
    const typename ompl::NearestNeighbors<_T>::DistanceFunction &df_;
  };

  struct ElemSortTo {
    ElemSortTo(const _T &e, const typename ompl::NearestNeighbors<_T>::DistanceFunction &df) : e_(e), df_(df) {}

    bool operator()(const _T &a, const _T &b) const {
        return df_(a, e_) < df_(b, e_);
    }

    const _T &e_;
    const typename ompl::NearestNeighbors<_T>::DistanceFunction &df_;
  };

  /** \brief The used distance function */
  typename ompl::NearestNeighbors<_T>::DistanceFunction euclDistFun_;
};

} /* namespace planning */

} /* namespace intel_wind */

#endif /* INTEL_WIND__MY_NEAREST_NEIGHBORS_HPP_ */
