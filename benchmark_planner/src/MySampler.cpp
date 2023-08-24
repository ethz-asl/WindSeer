/*!
 *  \file MySampler.cpp
 *  Allows sampling from an ellipsoid and also implements Dubins airplane adapted informed sampling.
 *
 *  Created on: Feb, 04 2016
 *      Author: Daniel Schneider, ASL
 *              Florian Achermann, ASL
 *
 *  Copy from OMPL (http://ompl.kavrakilab.org/classompl_1_1base_1_1PathLengthDirectInfSampler.html)!
 */

#include "MySampler.hpp"

#include <limits>

#include <boost/make_shared.hpp>
#include <ompl/util/Exception.h>
#include <ompl/base/OptimizationObjective.h>
#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/base/StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include "params.hpp"

namespace intel_wind {

namespace planning {

/////////////////////////////////////////////////////////////////////////////////////////////
//Public functions:

// The direct ellipsoid sampling class for path-length:
MySampler::MySampler(const ob::ProblemDefinitionPtr probDefn, unsigned int maxNumberCalls,
        double max_wind_x, double max_wind_y, double max_wind_z)
    : InformedSampler(probDefn, maxNumberCalls),
      summedMeasure_(0.0),
      informedIdx_(0u),
      uninformedIdx_(0u),
      stateValidityChecker_(probDefn->getSpaceInformation()->getStateValidityChecker()),
      zMaxStartGoal_(std::numeric_limits<double>::min()),
      zMinStartGoal_(std::numeric_limits<double>::max()),
      stateSpaceZMax_(std::numeric_limits<double>::max()),
      maxSpeed_(v_air_param + sqrt(max_wind_x*max_wind_x + max_wind_y*max_wind_y + max_wind_z*max_wind_z)),
      maxWindX_(max_wind_x),
      maxWindY_(max_wind_y),
      maxWindZ_(max_wind_z)
      {
  // Variables
  // The number of start states
  unsigned int numStarts;
  // The number of goal states
  unsigned numGoals;
  // The foci of the PHSs as a std::vector of states. Goals must be nonconst, as we need to allocate them (unfortunately):
  const ob::State* startState;
  ob::State* goalState;

  if (probDefn_->getGoal()->hasType(ompl::base::GOAL_SAMPLEABLE_REGION) == false) {
    throw ompl::Exception(
        "MySampler: The direct path-length informed sampler currently only supports goals that can be cast to a sampleable goal region (i.e., are countable sets).");
  }

  // Store the number of starts and goals
  numStarts = probDefn_->getStartStateCount();
  numGoals = probDefn_->getGoal()->as<ompl::base::GoalSampleableRegion>()->maxSampleCount();

  // Sanity check that there is atleast one of each
  if (numStarts < 1u || numGoals < 1u) {
    throw ompl::Exception(
        "MySampler: There must be at least 1 start and and 1 goal state when the informed sampler is created.");
  }

  if (numStarts != 1u) {
    OMPL_WARN("This sampler only supports one start state. The first is taken, all others are neglected.");
  }

  if (numGoals > 1u) {
    OMPL_INFORM("This sampler only samples one goal state although the goal region allows multiple samples.");
  }

  // Check that the provided statespace is compatible and extract the necessary indices.
  // The statespace must either be R^n or SE(2) or SE(3)
  if (InformedSampler::space_->isCompound() == false) {
    if (InformedSampler::space_->getType() == ob::STATE_SPACE_REAL_VECTOR) {
      informedIdx_ = 0u;
      uninformedIdx_ = 0u;
    } else {
      throw ompl::Exception("MySampler only supports RealVector, SE2 and SE3 StateSpaces.");
    }
  } else if (InformedSampler::space_->isCompound() == true) {
    // Check that it is SE2 or SE3
    if (InformedSampler::space_->getType() == ob::STATE_SPACE_SE2 || InformedSampler::space_->getType() == ob::STATE_SPACE_SE3) {
      // Variable:
      // An ease of use upcasted pointer to the space as a compound space
      const ob::CompoundStateSpace* compoundSpace = InformedSampler::space_->as<ob::CompoundStateSpace>();

      // Sanity check
      if (compoundSpace->getSubspaceCount() != 2u) {
        // Pout
        throw ompl::Exception("The provided compound StateSpace is SE(2) or SE(3) but does not have exactly 2 subspaces.");
      }

      // Iterate over the state spaces, finding the real vector and SO components.
      for (unsigned int idx = 0u; idx < InformedSampler::space_->as<ob::CompoundStateSpace>()->getSubspaceCount(); ++idx) {
        // Check if the space is real-vectored, SO2 or SO3
        if (compoundSpace->getSubspace(idx)->getType() == ob::STATE_SPACE_REAL_VECTOR) {
          informedIdx_ = idx;
        } else if (compoundSpace->getSubspace(idx)->getType() == ob::STATE_SPACE_SO2) {
          uninformedIdx_ = idx;
        } else if (compoundSpace->getSubspace(idx)->getType() == ob::STATE_SPACE_SO3) {
          uninformedIdx_ = idx;
        } else {
          // Pout
          throw ompl::Exception(
              "The provided compound StateSpace is SE(2) or SE(3) but contains a subspace that is not R^2, R^3, SO(2), or SO(3).");
        }
      }
    } else {
      throw ompl::Exception("MySampler only supports RealVector, SE2 and SE3 statespaces.");
    }
  }

  // Create a sampler for the whole space that we can use if we have no information
  baseSampler_ = InformedSampler::space_->allocDefaultStateSampler();

  // Check if the space is compound
  if (InformedSampler::space_->isCompound() == false) {
    // It is not.

    // The informed subspace is the full space
    informedSubSpace_ = InformedSampler::space_;

    // And the uniformed subspace and its associated sampler are null
    uninformedSubSpace_ = ob::StateSpacePtr();
    uninformedSubSampler_ = ob::StateSamplerPtr();
  } else {
    // It is

    // Get a pointer to the informed subspace...
    informedSubSpace_ = InformedSampler::space_->as<ob::CompoundStateSpace>()->getSubspace(informedIdx_);

    // And the uninformed subspace is the remainder.
    uninformedSubSpace_ = InformedSampler::space_->as<ob::CompoundStateSpace>()->getSubspace(uninformedIdx_);

    // Create a sampler for the uniformed subset:
    uninformedSubSampler_ = uninformedSubSpace_->allocDefaultStateSampler();
  }

  // Store the start focus
  startState = probDefn_->getStartState(0u);

  // Store the goal focus
  goalState = InformedSampler::space_->allocState();
  probDefn_->getGoal()->as<ompl::base::GoalSampleableRegion>()->sampleGoal(goalState);

  // Create PHS for the start/goal configuration
  startState_ = getInformedSubstate(startState);
  goalState_ = getInformedSubstate(goalState);

  phsPtr_ = boost::make_shared<ompl::ProlateHyperspheroid>(informedSubSpace_->getDimension(), &startState_[0],
      &goalState_[0]);

  // get min/max z value of start/goal
  zMinStartGoal_ = std::min(startState_[2], goalState_[2]);
  zMaxStartGoal_ = std::max(startState_[2], goalState_[2]);

  // get upper z bound of the state space
  std::vector<double> temp_state = goalState_;
  temp_state[2] = std::numeric_limits<double>::max();
  informedSubSpace_->copyFromReals(goalState->as<ob::CompoundState>()->components[informedIdx_], temp_state);
  InformedSampler::space_->enforceBounds(goalState);
  temp_state = getInformedSubstate(goalState);
  stateSpaceZMax_ = temp_state[2];

  // Finally deallocate the the goal state:
  InformedSampler::space_->freeState(goalState);
}


MySampler::~MySampler() {
}


bool MySampler::sampleUniform(ob::State *statePtr, const ob::Cost &maxCost) {
  unsigned int iter = 0u;

  //Call the sampleUniform helper function with my iteration counter:
  return sampleUniform(statePtr, maxCost, &iter);
}


bool MySampler::sampleUniform(ob::State *statePtr, const ob::Cost &minCost, const ob::Cost &maxCost) {
  // Sample from the larger PHS until the sample does not lie within the smaller PHS.
  // Since volume in a sphere/spheroid is proportionately concentrated near the surface, this isn't horribly inefficient, though a direct method would be better

  bool foundSample = false;

  // Spend numIters_ iterations trying to find an informed sample:
  for (unsigned int i = 0u; i < InformedSampler::numIters_ && foundSample == false; ++i) {
    foundSample = sampleUniform(statePtr, maxCost, &i);

    if (foundSample == true) {
      // Check that it meets the lower bound.
      ob::Cost sampledCost = heuristicSolnCost(statePtr);

      foundSample = InformedSampler::opt_->isCostEquivalentTo(minCost, sampledCost)
          || InformedSampler::opt_->isCostBetterThan(minCost, sampledCost);
    }
  }
  return foundSample;
}


bool MySampler::hasInformedMeasure() const {
  return true;
}


double MySampler::getInformedMeasure(const ob::Cost &currentCost) const {
  double informedMeasure = 0.0;

  double diam = currentCost.value() * maxSpeed_;
  if (diam < phsPtr_->getMinTransverseDiameter()) {
    // could actually happen because of numerical errors
    diam = phsPtr_->getMinTransverseDiameter();
  }

  informedMeasure = phsPtr_->getPhsMeasure(diam);

  // If the space is compound, further multiplied by the measure of the uniformed subspace
  if (InformedSampler::space_->isCompound() == true) {
    informedMeasure = informedMeasure * uninformedSubSpace_->getMeasure();
  }

  return std::min(InformedSampler::space_->getMeasure(), informedMeasure);
}


ob::Cost MySampler::heuristicSolnCost(const ob::State *statePtr) const {
  std::vector<double> rawData = getInformedSubstate(statePtr);

  const double dx1 = startState_[0] - rawData[0];
  const double dy1 = startState_[1] - rawData[1];
  const double dz1 = startState_[2] - rawData[2];

  const double dx2 = rawData[0] - goalState_[0];
  const double dy2 = rawData[1] - goalState_[1];
  const double dz2 = rawData[2] - goalState_[2];

  const double eucl_dist1 = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1);
  const double eucl_dist2 = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2);

  double dist1 = eucl_dist1;
  double dist2 = eucl_dist2;

  const double proj_wind_1 = fabs(dx1)/eucl_dist1 * maxWindX_ +
      fabs(dy1)/eucl_dist1 * maxWindY_ +
      fabs(dz1)/eucl_dist1 * maxWindZ_;

  const double proj_wind_2 = fabs(dx2)/eucl_dist2 * maxWindX_ +
      fabs(dy2)/eucl_dist2 * maxWindY_ +
      fabs(dz2)/eucl_dist2 * maxWindZ_;

  return ob::Cost(eucl_dist1 / (v_air_param + proj_wind_1) + eucl_dist2 / (v_air_param + proj_wind_2));
}
/////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////
//Private functions:
bool MySampler::sampleUniform(ob::State *statePtr, const ob::Cost &maxCost, unsigned int *iters) {
  bool foundSample = false;

  // Check if a solution path has been found
  if (InformedSampler::opt_->isFinite(maxCost) == false) {
    baseSampler_->sampleUniform(statePtr);

    ++(*iters);

    foundSample = true;
  } else  // We have a solution
  {
    // Update the definitions of the PHSs
    updatePhsDefinition(maxCost);

    // Sample from the PHSs.
    // When the summed measure of the PHSes are suitably large, it makes more sense to just sample from the entire planning space and keep the sample if it lies in any PHS
    // Check if the average measure is greater than half the domain's measure. Half is an arbitrary number.
    if (informedSubSpace_->getMeasure() * 0.5 < summedMeasure_) {
      // The measure is large, sample from the entire world and keep if it's in any PHS
      foundSample = sampleBoundsRejectPhs(statePtr, iters, maxCost);
    } else {
      // The measure is sufficiently small that we will directly sample the PHSes, with the weighting given by their relative measures
      foundSample = samplePhsRejectBounds(statePtr, iters, maxCost);
    }
  }

  if (foundSample)
    foundSample = getValidSampleByUpshifting(statePtr, maxCost);

  return foundSample;
}


bool MySampler::sampleBoundsRejectPhs(ob::State* statePtr, unsigned int *iters, const ob::Cost &maxCost) {
  bool foundSample = false;

  // Spend numIters_ iterations trying to find an informed sample:
  while (foundSample == false && *iters < InformedSampler::numIters_) {
    baseSampler_->sampleUniform(statePtr);

    foundSample = opt_->isCostBetterThan(heuristicSolnCost(statePtr), maxCost);

    ++(*iters);
  }
  return foundSample;
}


bool MySampler::samplePhsRejectBounds(ob::State *statePtr, unsigned int *iters, const ob::Cost &maxCost) {
  bool foundSample = false;

  while (foundSample == false && *iters < InformedSampler::numIters_) {
    std::vector<double> informedVector(informedSubSpace_->getDimension());

    // Use the PHS to get a sample in the informed subspace irrespective of boundary
    rng_.uniformProlateHyperspheroid(phsPtr_, &informedVector[0]);
    foundSample = true;

    if (foundSample == true) {
      createFullState(statePtr, informedVector);

      foundSample = InformedSampler::space_->satisfiesBounds(statePtr);
    }

    ++(*iters);
  }

  return foundSample;
}


std::vector<double> MySampler::getInformedSubstate(const ob::State *statePtr) const {
  std::vector<double> rawData(informedSubSpace_->getDimension());

  if (InformedSampler::space_->isCompound() == false) {
    informedSubSpace_->copyToReals(rawData, statePtr);
  } else {
    informedSubSpace_->copyToReals(rawData, statePtr->as<ob::CompoundState>()->components[informedIdx_]);
  }

  return rawData;
}


void MySampler::createFullState(ob::State * statePtr, const std::vector<double> &informedVector) {
  // If there is an extra "uninformed" subspace, we need to add that to the state before converting the raw vector representation into a state....
  if (InformedSampler::space_->isCompound() == false) {
    informedSubSpace_->copyFromReals(statePtr, informedVector);
  } else {
    ob::State *uninformedState = uninformedSubSpace_->allocState();

    informedSubSpace_->copyFromReals(statePtr->as<ob::CompoundState>()->components[informedIdx_], informedVector);

    uninformedSubSampler_->sampleUniform(uninformedState);

    uninformedSubSpace_->copyState(statePtr->as<ob::CompoundState>()->components[uninformedIdx_], uninformedState);

    uninformedSubSpace_->freeState(uninformedState);
  }
}


void MySampler::updatePhsDefinition(const ob::Cost &maxCost) {
  double transverse_diameter = maxCost.value();

  transverse_diameter *= maxSpeed_;

  if (phsPtr_->getMinTransverseDiameter() < transverse_diameter) {
    phsPtr_->setTransverseDiameter(transverse_diameter);

    summedMeasure_ = phsPtr_->getPhsMeasure();
  } else {
    phsPtr_->setTransverseDiameter(phsPtr_->getMinTransverseDiameter());

    // Set the summed measure to 0.0 (as a degenerate PHS is a line):
    summedMeasure_ = 0.0;
  }
}


bool MySampler::getValidSampleByUpshifting(ob::State *statePtr, const ob::Cost &maxCost) {
  const double height_shift = InformedSampler::space_->getLongestValidSegmentLength() * N_SHIFT;
  bool valid = stateValidityChecker_->isValid(statePtr);
  bool isInPhs = true;

  // try to get a valid sample
  std::vector<double> informedVector;
  if (InformedSampler::opt_->isFinite(maxCost) == false) {
    informedVector = getInformedSubstate(statePtr);
    // resample z if lower than zMinStartGoal and only geometric planning is done
    if (((not valid) || (informedVector[2] < zMinStartGoal_))) {
      informedVector[2] = rng_.uniformReal(zMinStartGoal_, zMaxStartGoal_);
      informedSubSpace_->copyFromReals(statePtr->as<ob::CompoundState>()->components[informedIdx_], informedVector);
      valid = stateValidityChecker_->isValid(statePtr);
    }

    while ((not valid) && (informedVector[2] < stateSpaceZMax_)) {
      // shift state
      informedVector = getInformedSubstate(statePtr);
      informedVector[2] += height_shift;
      informedSubSpace_->copyFromReals(statePtr->as<ob::CompoundState>()->components[informedIdx_], informedVector);

      valid = stateValidityChecker_->isValid(statePtr);
    }
  } else {
    informedVector = getInformedSubstate(statePtr);
    if (((not valid) || (informedVector[2] < zMinStartGoal_))) {
      informedVector[2] = rng_.uniformReal(zMinStartGoal_, zMaxStartGoal_);
      informedSubSpace_->copyFromReals(statePtr->as<ob::CompoundState>()->components[informedIdx_], informedVector);
      valid = stateValidityChecker_->isValid(statePtr);
      isInPhs = opt_->isCostBetterThan(heuristicSolnCost(statePtr), maxCost);
    }

    while ((not valid) && (informedVector[2] < stateSpaceZMax_) && isInPhs) {
      // shift state
      informedVector = getInformedSubstate(statePtr);
      informedVector[2] += height_shift;
      informedSubSpace_->copyFromReals(statePtr->as<ob::CompoundState>()->components[informedIdx_], informedVector);

      valid = stateValidityChecker_->isValid(statePtr);
      isInPhs = opt_->isCostBetterThan(heuristicSolnCost(statePtr), maxCost);
    }
  }
  return valid && isInPhs;
}

/////////////////////////////////////////////////////////////////////////////////////////////

} // namespace planning

} // namespace intel_wind
