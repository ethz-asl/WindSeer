/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Rice University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Rice University nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/
/*
 * This file is a copy of the standard ompl RRTstar.hpp.
 * It is extended to allow proper planning for non-symmetrical space and adding a solution path from another planner.
 * The solve, keepCondition, motionHeuristic functions are modified, the inGoal variable added to Motion.
 * The custom functions addSolutionPathToMotionTree and setFoundGeometricInitialSolution are added.
 */

/* Authors: Alejandro Perez, Sertac Karaman, Ryan Luna, Luis G. Torres, Ioan Sucan, Javier V Gomez, Jonathan Gammell, Florian Achermann, Daniel Schneider */

#include "MyRRTstar.hpp"

#include <vector>
#include <algorithm>
#include <limits>

#if(TIME_ENABLED)
  #include <chrono>
#endif

#include <boost/math/constants/constants.hpp>
#include <boost/make_shared.hpp>
#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/tools/config/SelfConfig.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/Goal.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/MotionValidator.h>
#include <ompl/base/goals/GoalState.h>
#include <ompl/util/GeometricEquations.h>
#include <ompl/base/samplers/InformedStateSampler.h>
#include <ompl/base/samplers/informed/RejectionInfSampler.h>

#include "MyNearestNeighbors.hpp"

namespace intel_wind {

namespace planning {

namespace ot = ompl::tools;


MyRRTstar::MyRRTstar(const ob::SpaceInformationPtr &si) :
        ob::Planner(si, "MyRRTstar"),
        goalBias_(0.05),
        maxDistance_(0.0),
        useKNearest_(true),
        rewireFactor_(1.1),
        k_rrg_(0u),
        r_rrg_(0.0),
        delayCC_(true),
        lastGoalMotion_(NULL),
        customNearestNeighbor_(false),
        computekNearestNeighbors2way_(false),
        useTreePruning_(false),
        pruneThreshold_(0.05),
        usePrunedMeasure_(false),
        useInformedSampling_(false),
        useRejectionSampling_(false),
        useNewStateRejection_(false),
        useAdmissibleCostToCome_(true),
        numSampleAttempts_ (100u),
        bestCost_(std::numeric_limits<double>::quiet_NaN()),
        prunedCost_(std::numeric_limits<double>::quiet_NaN()),
        prunedMeasure_(0.0),
        iterations_(0u),
        foundGeometricInitialSolution_(false) {
  specs_.approximateSolutions = true;
  specs_.optimizingPaths = true;
  specs_.canReportIntermediateSolutions = true;

  Planner::declareParam<double>("range", this, &MyRRTstar::setRange, &MyRRTstar::getRange, "0.:1.:10000.");
  Planner::declareParam<double>("goal_bias", this, &MyRRTstar::setGoalBias, &MyRRTstar::getGoalBias, "0.:.05:1.");
  Planner::declareParam<double>("rewire_factor", this, &MyRRTstar::setRewireFactor, &MyRRTstar::getRewireFactor, "1.0:0.01:2.0");
  Planner::declareParam<bool>("use_k_nearest", this, &MyRRTstar::setKNearest, &MyRRTstar::getKNearest, "0,1");
  Planner::declareParam<bool>("delay_collision_checking", this, &MyRRTstar::setDelayCC, &MyRRTstar::getDelayCC, "0,1");
  Planner::declareParam<bool>("tree_pruning", this, &MyRRTstar::setTreePruning, &MyRRTstar::getTreePruning, "0,1");
  Planner::declareParam<double>("prune_threshold", this, &MyRRTstar::setPruneThreshold, &MyRRTstar::getPruneThreshold, "0.:.01:1.");
  Planner::declareParam<bool>("pruned_measure", this, &MyRRTstar::setPrunedMeasure, &MyRRTstar::getPrunedMeasure, "0,1");
  Planner::declareParam<bool>("informed_sampling", this, &MyRRTstar::setInformedSampling, &MyRRTstar::getInformedSampling, "0,1");
  Planner::declareParam<bool>("sample_rejection", this, &MyRRTstar::setSampleRejection, &MyRRTstar::getSampleRejection, "0,1");
  Planner::declareParam<bool>("new_state_rejection", this, &MyRRTstar::setNewStateRejection, &MyRRTstar::getNewStateRejection, "0,1");
  Planner::declareParam<bool>("use_admissible_heuristic", this, &MyRRTstar::setAdmissibleCostToCome, &MyRRTstar::getAdmissibleCostToCome, "0,1");
  Planner::declareParam<bool>("focus_search", this, &MyRRTstar::setFocusSearch, &MyRRTstar::getFocusSearch, "0,1");
  Planner::declareParam<unsigned int>("number_sampling_attempts", this, &MyRRTstar::setNumSamplingAttempts, &MyRRTstar::getNumSamplingAttempts, "10:10:100000");
  Planner::declareParam<bool>("custom_nearest_neighbor", this, &MyRRTstar::setCustomNearestNeighbor, &MyRRTstar::getCustomNearestNeighbor, "0,1");
  Planner::declareParam<bool>("compute_k_nearest_neighbors_2way", this, &MyRRTstar::setComputekNearestNeighbors2way, &MyRRTstar::getComputekNearestNeighbors2way, "0,1");

  addPlannerProgressProperty("iterations INTEGER",
      boost::bind(&MyRRTstar::numIterationsProperty, this));
  addPlannerProgressProperty("best cost REAL",
      boost::bind(&MyRRTstar::bestCostProperty, this));
}


MyRRTstar::~MyRRTstar() {
  freeMemory();
}


void MyRRTstar::getPlannerData(ob::PlannerData &data) const {
  Planner::getPlannerData(data);

  std::vector<Motion*> motions;
  if (nn_)
    nn_->list(motions);

  if (lastGoalMotion_)
    data.addGoalVertex(ob::PlannerDataVertex(lastGoalMotion_->state));

  for (std::size_t i = 0 ; i < motions.size() ; ++i) {
    if (motions[i]->parent == NULL)
      data.addStartVertex(ob::PlannerDataVertex(motions[i]->state));
    else
      data.addEdge(ob::PlannerDataVertex(motions[i]->parent->state),
          ob::PlannerDataVertex(motions[i]->state));
  }
}


ob::PlannerStatus MyRRTstar::solve(const ob::PlannerTerminationCondition &ptc) {
  checkValidity();
  ob::Goal                  *goal   = pdef_->getGoal().get();
  ob::GoalSampleableRegion  *goal_s = dynamic_cast<ob::GoalSampleableRegion*>(goal);

  const bool symCost = opt_->isSymmetric();
  const bool symStateSpace = si_->getStateSpace()->hasSymmetricDistance() && si_->getStateSpace()->hasSymmetricInterpolate();

  // Check if there are more starts
  if (!foundGeometricInitialSolution_) {
    if (pis_.haveMoreStartStates()) {
      // There are, add them
      while (const ob::State *st = pis_.nextStart()) {
        Motion *motion = new Motion(si_);
        si_->copyState(motion->state, st);
        motion->cost = opt_->identityCost();
        nn_->add(motion);
        startMotions_.push_back(motion);
      }

      // And assure that, if we're using an informed sampler, it's reset
      infSampler_.reset();
    }
  }

  if (nn_->size() == 0) {
    OMPL_ERROR("%s: There are no valid initial states!", getName().c_str());
    return ob::PlannerStatus::INVALID_START;
  }

  //Allocate a sampler if necessary
  if (!sampler_ && !infSampler_) {
    allocSampler();
  }

  OMPL_INFORM("%s: Starting planning with %u states already in datastructure", getName().c_str(), nn_->size());

  if ((useTreePruning_ || useRejectionSampling_ || useInformedSampling_ || useNewStateRejection_) && !si_->getStateSpace()->isMetricSpace())
    OMPL_WARN("%s: The state space (%s) is not metric and as a result the optimization objective may not satisfy the triangle inequality. "
        "You may need to disable pruning or rejection."
        , getName().c_str(), si_->getStateSpace()->getName().c_str());

  const ob::ReportIntermediateSolutionFn intermediateSolutionCallback = pdef_->getIntermediateSolutionCallback();

  Motion *solution       = lastGoalMotion_;

  Motion *approximation  = NULL;
  double approximatedist = std::numeric_limits<double>::infinity();
  bool sufficientlyShort = false;

  Motion *rmotion        = new Motion(si_);
  ob::State *rstate    = rmotion->state;
  ob::State *xstate    = si_->allocState();

  std::vector<Motion*>       nbh;

  std::vector<ob::Cost>    costs;
  std::vector<ob::Cost>    incCosts;
  std::vector<std::size_t>   sortedCostIndices;

  std::vector<int>           valid;
  unsigned int               rewireTest = 0;
  unsigned int               statesGenerated = 0;

  if (solution)
    OMPL_INFORM("%s: Starting planning with existing solution of cost %.5f", getName().c_str(), solution->cost.value());

  if (useKNearest_)
    OMPL_INFORM("%s: Initial k-nearest value of %u", getName().c_str(),
        (unsigned int)std::ceil(k_rrg_ * log((double)(nn_->size() + 1u))));
  else
    OMPL_INFORM("%s: Initial rewiring radius of %.2f", getName().c_str(),
        std::min(maxDistance_, r_rrg_*std::pow(log((double)(nn_->size() + 1u))/
            ((double)(nn_->size() + 1u)), 1/(double)(si_->getStateDimension()))));

  // our functor for sorting nearest neighbors
  CostIndexCompare compareFn(costs, *opt_);

  while (ptc == false && opt_->isSatisfied(bestCost_) == false) {
    iterations_++;

    // sample random state (with goal biasing)
    // Goal samples are only sampled until maxSampleCount() goals are in the tree, to prohibit duplicate goal states.
    if (goal_s && goalMotions_.size() < goal_s->maxSampleCount() && rng_.uniform01() < goalBias_ && goal_s->canSample())
      goal_s->sampleGoal(rstate);
    else {
      // Attempt to generate a sample, if we fail (e.g., too many rejection attempts), skip the remainder of this loop and return to try again
      if (!sampleUniform(rstate)) {
        continue;
      }
    }

    // find closest state in the tree
    Motion *nmotion = nn_->nearest(rmotion);

    if (intermediateSolutionCallback && si_->equalStates(nmotion->state, rstate)) {
      continue;
    }

    ob::State *dstate = rstate;

    // find state to add to the tree
    double d = si_->distance(nmotion->state, rstate);

    if (std::isnan(d)) {
      continue;
    }

    if (d > maxDistance_) {
      si_->getStateSpace()->interpolate(nmotion->state, rstate, maxDistance_ / d, xstate);
      dstate = xstate;
    }

    // Check if the motion between the nearest state and the state to add is valid
    if (si_->checkMotion(nmotion->state, dstate) && !std::isinf(opt_->motionCost(nmotion->state, dstate).value())) {
      // create a motion
      Motion *motion = new Motion(si_);
      si_->copyState(motion->state, dstate);
      motion->parent = nmotion;
      motion->incCost = opt_->motionCost(nmotion->state, motion->state);
      motion->cost = opt_->combineCosts(nmotion->cost, motion->incCost);

      // Find nearby neighbors of the new motion
      getNeighbors(motion, nbh);

      rewireTest += nbh.size();
      ++statesGenerated;

      // cache for distance computations
      //
      // Our cost caches only increase in size, so they're only
      // resized if they can't fit the current neighborhood
      if (costs.size() < nbh.size()) {
        costs.resize(nbh.size());
        incCosts.resize(nbh.size());
        sortedCostIndices.resize(nbh.size());
      }

      // cache for motion validity (only useful in a symmetric space)
      //
      // Our validity caches only increase in size, so they're
      // only resized if they can't fit the current neighborhood
      if (valid.size() < nbh.size())
        valid.resize(nbh.size());
      std::fill(valid.begin(), valid.begin() + nbh.size(), 0);

      // Finding the nearest neighbor to connect to
      // By default, neighborhood states are sorted by cost, and collision checking
      // is performed in increasing order of cost
      if (delayCC_) {
        // calculate all costs and distances
        for (std::size_t i = 0; i < nbh.size(); ++i) {
          incCosts[i] = opt_->motionCost(nbh[i]->state, motion->state);
          costs[i] = opt_->combineCosts(nbh[i]->cost, incCosts[i]);
        }

        // sort the nodes
        //
        // we're using index-value pairs so that we can get at
        // original, unsorted indices
        for (std::size_t i = 0; i < nbh.size(); ++i)
          sortedCostIndices[i] = i;
        std::sort(sortedCostIndices.begin(), sortedCostIndices.begin() + nbh.size(), compareFn);

        // collision check until a valid motion is found
        //
        // ASYMMETRIC CASE: it's possible that none of these
        // neighbors are valid. This is fine, because motion
        // already has a connection to the tree through
        // nmotion (with populated cost fields!).
        for (std::vector<std::size_t>::const_iterator i = sortedCostIndices.begin();
            i != sortedCostIndices.begin() + nbh.size(); ++i)
        {
          if (std::isnan(incCosts[*i].value()))
            valid[*i] = -1;
          else {
            if (nbh[*i] == nmotion || si_->checkMotion(nbh[*i]->state, motion->state)) {
              motion->incCost = incCosts[*i];
              motion->cost = costs[*i];
              motion->parent = nbh[*i];
              valid[*i] = 1;
              break;
            } else
              valid[*i] = -1;
          }
        }
      } else { // if not delayCC
        motion->incCost = opt_->motionCost(nmotion->state, motion->state);
        motion->cost = opt_->combineCosts(nmotion->cost, motion->incCost);
        // find which one we connect the new state to
        for (std::size_t i = 0 ; i < nbh.size(); ++i) {
          if (nbh[i] != nmotion) {
            incCosts[i] = opt_->motionCost(nbh[i]->state, motion->state);
            costs[i] = opt_->combineCosts(nbh[i]->cost, incCosts[i]);
            if (opt_->isCostBetterThan(costs[i], motion->cost)) {
              if (si_->checkMotion(nbh[i]->state, motion->state) && !std::isinf(costs[i].value())) {
                motion->incCost = incCosts[i];
                motion->cost = costs[i];
                motion->parent = nbh[i];
                valid[i] = 1;
              } else
                valid[i] = -1;
            }
          } else {
            incCosts[i] = motion->incCost;
            costs[i] = motion->cost;
            valid[i] = 1;
          }
        }
      }

      if (useNewStateRejection_) {
        if (opt_->isCostBetterThan(solutionHeuristic(motion), bestCost_)) {
          nn_->add(motion);
          motion->parent->children.push_back(motion);
        } else { // If the new motion does not improve the best cost it is ignored.
          si_->freeState(motion->state);
          delete motion;
          continue;
        }
      } else {
        // add motion to the tree
        nn_->add(motion);
        motion->parent->children.push_back(motion);
      }

      // get the correct nbh
      if (useKNearest_ && customNearestNeighbor_ && computekNearestNeighbors2way_) {
        const double cardDbl = static_cast<double>(nn_->size());
        const unsigned int k = std::ceil(k_rrg_ * log(cardDbl));
        boost::static_pointer_cast<MyNearestNeighbors<Motion*>>(nn_)
            ->nearestKFrom(motion, k, nbh);
      }

      bool checkForSolution = false;
      for (std::size_t i = 0, end = nbh.size(); i < end; ++i) {
        if (nbh[i] != motion->parent) {
          ob::Cost nbhIncCost;
          if (symCost)
            nbhIncCost = incCosts[i];
          else
            nbhIncCost = opt_->motionCost(motion->state, nbh[i]->state);
          ob::Cost nbhNewCost = opt_->combineCosts(motion->cost, nbhIncCost);
          if (opt_->isCostBetterThan(nbhNewCost, nbh[i]->cost)) {
            bool motionValid;
            if (valid[i] == 0 || !symStateSpace) {
              motionValid = si_->checkMotion(motion->state, nbh[i]->state) && !std::isinf(nbhNewCost.value());
            } else {
              motionValid = (valid[i] == 1);
            }

            if (motionValid) {
              // Remove this node from its parent list
              removeFromParent(nbh[i]);

              // Add this node to the new parent
              nbh[i]->parent = motion;
              nbh[i]->incCost = nbhIncCost;
              nbh[i]->cost = nbhNewCost;
              nbh[i]->parent->children.push_back(nbh[i]);

              // Update the costs of the node's children
              updateChildCosts(nbh[i]);

              checkForSolution = true;
            }
          }
        }
      }

      // Add the new motion to the goalMotion_ list, if it satisfies the goal
      double distanceFromGoal;
      if (goal->isSatisfied(motion->state, &distanceFromGoal)) {
        motion->inGoal = true;
        goalMotions_.push_back(motion);
        checkForSolution = true;
      }

      // Checking for solution or iterative improvement
      if (checkForSolution) {
        bool updatedSolution = false;
        for (size_t i = 0, end = goalMotions_.size(); i < end; ++i) {
          if (opt_->isCostBetterThan(goalMotions_[i]->cost, bestCost_)) {
            if (opt_->isFinite(bestCost_) == false) {
              OMPL_INFORM("%s: Found an initial solution with a cost of %.2f in %u iterations (%u vertices in the graph)",
                  getName().c_str(), goalMotions_[i]->cost.value(), iterations_, nn_->size());
            }
            bestCost_ = goalMotions_[i]->cost;
            updatedSolution = true;
          }

          sufficientlyShort = opt_->isSatisfied(goalMotions_[i]->cost);
          if (sufficientlyShort) {
            solution = goalMotions_[i];
            break;
          } else if (!solution || opt_->isCostBetterThan(goalMotions_[i]->cost,solution->cost)) {
            solution = goalMotions_[i];
            updatedSolution = true;
          }
        }

        if (updatedSolution) {
          if (useTreePruning_) {
            pruneTree(bestCost_);
          }

          if (intermediateSolutionCallback) {
            std::vector<const ob::State *> spath;
            Motion *intermediate_solution = solution->parent; // Do not include goal state to simplify code.

            //Push back until we find the start, but not the start itself
            while (intermediate_solution->parent != NULL) {
              spath.push_back(intermediate_solution->state);
              intermediate_solution = intermediate_solution->parent;
            }

            intermediateSolutionCallback(this, spath, bestCost_);
          }
        }
      }

      // Checking for approximate solution (closest state found to the goal)
      if (goalMotions_.size() == 0 && distanceFromGoal < approximatedist) {
        approximation = motion;
        approximatedist = distanceFromGoal;
      }
    } else {
    }

    // terminate if a sufficient solution is found
    if (solution && sufficientlyShort)
      break;
  } // while (ptc == false && opt_->isSatisfied(bestCost_) == false)

  const bool approximate = (solution == NULL);
  bool addedSolution = false;
  if (approximate)
    solution = approximation;
  else
    lastGoalMotion_ = solution;

  if (solution != NULL) {
    ptc.terminate();
    // construct the solution path
    std::vector<Motion*> mpath;
    while (solution != NULL) {
      mpath.push_back(solution);
      solution = solution->parent;
    }

    // set the solution path
    og::PathGeometric *geoPath = new og::PathGeometric(si_);
    for (int i = mpath.size() - 1; i >= 0; --i)
      geoPath->append(mpath[i]->state);

    ob::PathPtr path(geoPath);
    // Add the solution path.
    ob::PlannerSolution psol(path);
    psol.setPlannerName(getName());
    if (approximate)
      psol.setApproximate(approximatedist);
    // Does the solution satisfy the optimization objective?
    psol.setOptimized(opt_, bestCost_, sufficientlyShort);
    pdef_->addSolutionPath(psol);

    addedSolution = true;
  }

  si_->freeState(xstate);
  if (rmotion->state)
    si_->freeState(rmotion->state);
  delete rmotion;

  OMPL_INFORM("%s: Created %u new states. Checked %u rewire options. %u goal states in tree. Final solution cost %.3f",
      getName().c_str(), statesGenerated, rewireTest, goalMotions_.size(), bestCost_.value());

  return ob::PlannerStatus(addedSolution, approximate);
}


void MyRRTstar::clear() {
  setup_ = false;
  Planner::clear();
  sampler_.reset();
  infSampler_.reset();
  freeMemory();
  if (nn_)
    nn_->clear();

  lastGoalMotion_ = NULL;
  goalMotions_.clear();
  startMotions_.clear();

  iterations_ = 0;
  bestCost_ = ob::Cost(std::numeric_limits<double>::quiet_NaN());
  prunedCost_ = ob::Cost(std::numeric_limits<double>::quiet_NaN());
  prunedMeasure_ = 0.0;
}


void MyRRTstar::setup() {
  Planner::setup();
  ot::SelfConfig sc(si_, getName());
  sc.configurePlannerRange(maxDistance_);
  if (!si_->getStateSpace()->hasSymmetricDistance() || !si_->getStateSpace()->hasSymmetricInterpolate()) {
    OMPL_WARN("%s requires a state space with symmetric distance and symmetric interpolation.", getName().c_str());
  }

  if (customNearestNeighbor_) {
    nn_.reset(new MyNearestNeighbors<Motion*>());
    boost::static_pointer_cast<MyNearestNeighbors<Motion*>>(nn_)
            ->setEuclDistanceFunction(boost::bind(&MyRRTstar::distanceFunction, this, _1, _2));
  } else {
    if (!nn_)
      nn_.reset(new ompl::NearestNeighborsLinear<Motion*>());
  }
  nn_->setDistanceFunction(boost::bind(&MyRRTstar::distanceFunction, this, _1, _2));

  // Setup optimization objective
  //
  // If no optimization objective was specified, then default to
  // optimizing path length as computed by the distance() function
  // in the state space.
  if (pdef_) {
    if (pdef_->hasOptimizationObjective())
      opt_ = pdef_->getOptimizationObjective();
    else {
      OMPL_INFORM("%s: No optimization objective specified. Defaulting to optimizing path length for the allowed planning time.", getName().c_str());
      opt_.reset(new ob::PathLengthOptimizationObjective(si_));

      // Store the new objective in the problem def'n
      pdef_->setOptimizationObjective(opt_);
    }
  } else {
    OMPL_INFORM("%s: problem definition is not set, deferring setup completion...", getName().c_str());
    setup_ = false;
  }

  // Get the measure of the entire space:
  prunedMeasure_ = si_->getSpaceMeasure();

  // Calculate some constants:
  calculateRewiringLowerBounds();

  // Set the bestCost_ and prunedCost_ as infinite
  bestCost_ = opt_->infiniteCost();
  prunedCost_ = opt_->infiniteCost();
}


void MyRRTstar::setTreePruning(const bool prune) {
  if (static_cast<bool>(opt_) == true) {
    if (opt_->hasCostToGoHeuristic() == false) {
      OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
    }
  }

  // If we just disabled tree pruning, but we are using prunedMeasure, we need to disable that as it required myself
  if (prune == false && getPrunedMeasure() == true) {
    setPrunedMeasure(false);
  }

  // Store
  useTreePruning_ = prune;
}


void MyRRTstar::setPrunedMeasure(bool informedMeasure) {
  if (static_cast<bool>(opt_) == true) {
    if (opt_->hasCostToGoHeuristic() == false) {
      OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
    }
  }

  // This option only works with informed sampling
  if (informedMeasure == true && (useInformedSampling_ == false || useTreePruning_ == false)) {
    OMPL_ERROR("%s: InformedMeasure requires InformedSampling and TreePruning.", getName().c_str());
  }

  // Check if we're changed and update parameters if we have:
  if (informedMeasure != usePrunedMeasure_) {
    // Store the setting
    usePrunedMeasure_ = informedMeasure;

    // Update the prunedMeasure_ appropriately, if it has been configured.
    if (setup_ == true) {
      if (usePrunedMeasure_) {
        prunedMeasure_ = infSampler_->getInformedMeasure(prunedCost_);
      } else {
        prunedMeasure_ = si_->getSpaceMeasure();
      }
    }

    // And either way, update the rewiring radius if necessary
    if (useKNearest_ == false) {
      calculateRewiringLowerBounds();
    }
  }
}


void MyRRTstar::setInformedSampling(bool informedSampling) {
  if (static_cast<bool>(opt_) == true) {
    if (opt_->hasCostToGoHeuristic() == false) {
      OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
    }
  }

  // This option is mutually exclusive with setSampleRejection, assert that:
  if (informedSampling == true && useRejectionSampling_ == true) {
    OMPL_ERROR("%s: InformedSampling and SampleRejection are mutually exclusive options.", getName().c_str());
  }

  // If we just disabled tree pruning, but we are using prunedMeasure, we need to disable that as it required myself
  if (informedSampling == false && getPrunedMeasure() == true) {
    setPrunedMeasure(false);
  }

  // Check if we're changing the setting of informed sampling. If we are, we will need to create a new sampler, which we only want to do if one is already allocated.
  if (informedSampling != useInformedSampling_) {
    //If we're disabled informedSampling, and prunedMeasure is enabled, we need to disable that
    if (informedSampling == false && usePrunedMeasure_ == true) {
      setPrunedMeasure(false);
    }

    // Store the value
    useInformedSampling_ = informedSampling;

    // If we currently have a sampler, we need to make a new one
    if (sampler_ || infSampler_) {
      // Reset the samplers
      sampler_.reset();
      infSampler_.reset();

      // Create the sampler
      allocSampler();
    }
  }
}


void MyRRTstar::setSampleRejection(const bool reject) {
  if (static_cast<bool>(opt_) == true) {
    if (opt_->hasCostToGoHeuristic() == false) {
      OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
    }
  }

  // This option is mutually exclusive with setSampleRejection, assert that:
  if (reject == true && useInformedSampling_ == true) {
    OMPL_ERROR("%s: InformedSampling and SampleRejection are mutually exclusive options.", getName().c_str());
  }

  // Check if we're changing the setting of rejection sampling. If we are, we will need to create a new sampler, which we only want to do if one is already allocated.
  if (reject != useRejectionSampling_) {
    // Store the setting
    useRejectionSampling_ = reject;

    // If we currently have a sampler, we need to make a new one
    if (sampler_ || infSampler_) {
      // Reset the samplers
      sampler_.reset();
      infSampler_.reset();

      // Create the sampler
      allocSampler();
    }
  }
}


void MyRRTstar::allocSampler() {
  // Allocate the appropriate type of sampler.
  if (useInformedSampling_) {
    // We are using informed sampling, this can end-up reverting to rejection sampling in some cases
    OMPL_INFORM("%s: Using informed sampling.", getName().c_str());
    infSampler_ = opt_->allocInformedStateSampler(pdef_, numSampleAttempts_);
  } else if (useRejectionSampling_) {
    // We are explicitly using rejection sampling.
    OMPL_INFORM("%s: Using rejection sampling.", getName().c_str());
    infSampler_ = boost::make_shared<ob::RejectionInfSampler>(pdef_, numSampleAttempts_);
  } else {
    // We are using a regular sampler
    sampler_ = si_->allocStateSampler();
  }
}


bool MyRRTstar::sampleUniform(ob::State *statePtr) {
  // Use the appropriate sampler
  if (useInformedSampling_ || useRejectionSampling_) {
    // Attempt the focused sampler and return the result.
    // If bestCost is changing a lot by small amounts, this could
    // be prunedCost_ to reduce the number of times the informed sampling
    // transforms are recalculated.
    return infSampler_->sampleUniform(statePtr, bestCost_);
  } else {
    // Simply return a state from the regular sampler
    sampler_->sampleUniform(statePtr);

    // Always true
    return true;
  }
}


void MyRRTstar::freeMemory() {
  if (nn_) {
    std::vector<Motion*> motions;
    nn_->list(motions);
    for (std::size_t i = 0 ; i < motions.size() ; ++i) {
      if (motions[i]->state)
        si_->freeState(motions[i]->state);
      delete motions[i];
    }
  }
}


void MyRRTstar::getNeighbors(Motion *motion, std::vector<Motion*> &nbh) const {
  double cardDbl = static_cast<double>(nn_->size() + 1u);
  if (useKNearest_) {
    //- k-nearest RRT*
    unsigned int k = std::ceil(k_rrg_ * log(cardDbl));
    nn_->nearestK(motion, k, nbh);
  } else {
    double r = std::min(maxDistance_, r_rrg_ * std::pow(log(cardDbl) / cardDbl, 1 / static_cast<double>(si_->getStateDimension())));
    nn_->nearestR(motion, r, nbh);
  }
}


void MyRRTstar::removeFromParent(Motion *m) {
  for (std::vector<Motion*>::iterator it = m->parent->children.begin ();
      it != m->parent->children.end (); ++it) {
    if (*it == m) {
      m->parent->children.erase(it);
      break;
    }
  }
}


void MyRRTstar::updateChildCosts(Motion *m) {
  for (std::size_t i = 0; i < m->children.size(); ++i) {
    m->children[i]->cost = opt_->combineCosts(m->cost, m->children[i]->incCost);
    updateChildCosts(m->children[i]);
  }
}


int MyRRTstar::pruneTree(const ob::Cost& pruneTreeCost) {
  // Variable
  // The percent improvement (expressed as a [0,1] fraction) in cost
  double fracBetter;
  // The number pruned
  int numPruned = 0;

  if (opt_->isFinite(prunedCost_)) {
    fracBetter = std::abs((pruneTreeCost.value() - prunedCost_.value())/prunedCost_.value());
  } else {
    fracBetter = 1.0;
  }

  if (fracBetter > pruneThreshold_) {
    // We are only pruning motions if they, AND all descendents, have a estimated cost greater than pruneTreeCost
    // The easiest way to do this is to find leaves that should be pruned and ascend up their ancestry until a motion is found that is kept.
    // To avoid making an intermediate copy of the NN structure, we process the tree by descending down from the start(s).
    // In the first pass, all Motions with a cost below pruneTreeCost, or Motion's with children with costs below pruneTreeCost are added to the replacement NN structure,
    // while all other Motions are stored as either a 'leaf' or 'chain' Motion. After all the leaves are disconnected and deleted, we check
    // if any of the the chain Motions are now leaves, and repeat that process until done.
    // This avoids (1) copying the NN structure into an intermediate variable and (2) the use of the expensive NN::remove() method.

    // Variable
    // The queue of Motions to process:
    std::queue<Motion*, std::deque<Motion*> > motionQueue;
    // The list of leaves to prune
    std::queue<Motion*, std::deque<Motion*> > leavesToPrune;
    // The list of chain vertices to recheck after pruning
    std::list<Motion*> chainsToRecheck;

    //Clear the NN structure:
    nn_->clear();

    // Put all the starts into the NN structure and their children into the queue:
    // We do this so that start states are never pruned.
    for (unsigned int i = 0u; i < startMotions_.size(); ++i) {
      // Add to the NN
      nn_->add(startMotions_.at(i));

      // Add their children to the queue:
      addChildrenToList(&motionQueue, startMotions_.at(i));
    }

    while (motionQueue.empty() == false) {
      // Test, can the current motion ever provide a better solution?
      if (keepCondition(motionQueue.front(), pruneTreeCost)) {
        // Yes it can, so it definitely won't be pruned
        // Add it back into the NN structure
        nn_->add(motionQueue.front());

        //Add it's children to the queue
        addChildrenToList(&motionQueue, motionQueue.front());
      } else {
        // No it can't, but does it have children?
        if (motionQueue.front()->children.empty() == false) {
          // Yes it does.
          // We can minimize the number of intermediate chain motions if we check their children
          // If any of them won't be pruned, then this motion won't either. This intuitively seems
          // like a nice balance between following the descendents forever.

          // Variable
          // Whether the children are definitely to be kept.
          bool keepAChild = false;

          // Find if any child is definitely not being pruned.
          for (unsigned int i = 0u; keepAChild == false && i < motionQueue.front()->children.size(); ++i) {
            // Test if the child can ever provide a better solution
            keepAChild = keepCondition(motionQueue.front()->children.at(i), pruneTreeCost);
          }

          // Are we *definitely* keeping any of the children?
          if (keepAChild) {
            // Yes, we are, so we are not pruning this motion
            // Add it back into the NN structure.
            nn_->add(motionQueue.front());
          } else {
            // No, we aren't. This doesn't mean we won't though
            // Move this Motion to the temporary list
            chainsToRecheck.push_back(motionQueue.front());
          }

          // Either way. add it's children to the queue
          addChildrenToList(&motionQueue, motionQueue.front());
        } else {
          // No, so we will be pruning this motion:
          leavesToPrune.push(motionQueue.front());
        }
      }

      // Pop the iterator, std::list::erase returns the next iterator
      motionQueue.pop();
    }

    // We now have a list of Motions to definitely remove, and a list of Motions to recheck
    // Iteratively check the two lists until there is nothing to to remove
    while (leavesToPrune.empty() == false) {
      // First empty the leave-to-prune
      while (leavesToPrune.empty() == false) {
        // If this leaf is a goal, remove it from the goal set
        if (leavesToPrune.front()->inGoal == true) {
          // Remove it
          goalMotions_.erase(std::remove(goalMotions_.begin(), goalMotions_.end(), leavesToPrune.front()), goalMotions_.end());
        }

        // Remove the leaf from its parent
        removeFromParent(leavesToPrune.front());

        // Erase the actual motion
        // First free the state
        si_->freeState(leavesToPrune.front()->state);

        // then delete the pointer
        delete leavesToPrune.front();

        // And finally remove it from the list, erase returns the next iterator
        leavesToPrune.pop();

        // Update our counter
        ++numPruned;
      }

      // Now, we need to go through the list of chain vertices and see if any are now leaves
      std::list<Motion*>::iterator mIter = chainsToRecheck.begin();
      while (mIter != chainsToRecheck.end()) {
        // Is the Motion a leaf?
        if ((*mIter)->children.empty() == true) {
          // It is, add to the removal queue
          leavesToPrune.push(*mIter);

          // Remove from this queue, getting the next
          mIter = chainsToRecheck.erase(mIter);
        } else {
          // Is isn't, skip to the next
          ++mIter;
        }
      }
    }

    // Now finally add back any vertices left in chainsToReheck.
    // These are chain vertices that have descendents that we want to keep
    for (std::list<Motion*>::const_iterator mIter = chainsToRecheck.begin(); mIter != chainsToRecheck.end(); ++mIter) {
      // Add the motion back to the NN struct:
      nn_->add(*mIter);
    }

    // All done pruning.
    // Update the cost at which we've pruned:
    prunedCost_ = pruneTreeCost;

    // And if we're using the pruned measure, the measure to which we've pruned
    if (usePrunedMeasure_) {
      prunedMeasure_ = infSampler_->getInformedMeasure(prunedCost_);

      if (useKNearest_ == false) {
        calculateRewiringLowerBounds();
      }
    }
    //No else, prunedMeasure_ is the si_ measure by default.
  }

  return numPruned;
}


ob::Cost MyRRTstar::solutionHeuristic(const Motion *motion) const {
  ob::Cost costToCome;
  if (useAdmissibleCostToCome_) {
    // Start with infinite cost
    costToCome = opt_->infiniteCost();

    //Find the min from each start
    for (unsigned int i = 0u; i < startMotions_.size(); ++i) {
      costToCome = opt_->betterCost(costToCome, opt_->motionCostHeuristic(startMotions_.at(i)->state, motion->state)); // lower-bounding cost from the start to the state
    }
  } else {
    costToCome = motion->cost; // current cost from the state to the goal
  }

  const ob::Cost costToGo = opt_->costToGo(motion->state, pdef_->getGoal().get()); // lower-bounding cost from the state to the goal
  return opt_->combineCosts(costToCome, costToGo); // add the two costs
}


void MyRRTstar::addChildrenToList(std::queue<Motion*, std::deque<Motion*> > *motionList, Motion* motion) {
  for (unsigned int j = 0u; j < motion->children.size(); ++j) {
    motionList->push(motion->children.at(j));
  }
}


bool MyRRTstar::keepCondition(const Motion* motion, const ob::Cost& threshold) const {
  // We keep if the cost-to-come-heuristic of motion is <= threshold, by checking
  // if (!threshold < heuristic), as if b is not better than a, then a is better than, or equal to, b

  // WARNING
  // The threshold value is encreased by a small margin to avoid deleting a goal state in which the path to the
  // goal state consists only of high altitude case segments. Due to floating point inaccuracies the lenth of
  // such a path might be bigger (around 1e-5) than the direct case, although the distance should be equal.
  // If a SEGFAULT at the end of the planning process happens this margin might need to be increased.
  // WARNING
  ob::Cost threshold_safe(threshold.value() + 1e-3);
  return !opt_->isCostBetterThan(threshold_safe, solutionHeuristic(motion));
}


void MyRRTstar::calculateRewiringLowerBounds() {
  double dimDbl = static_cast<double>(si_->getStateDimension());

  // k_rrg > e+e/d.  K-nearest RRT*
  k_rrg_ = rewireFactor_ * (boost::math::constants::e<double>() + (boost::math::constants::e<double>() / dimDbl));

  // r_rrg > 2*(1+1/d)^(1/d)*(measure/ballvolume)^(1/d)
  // If we're not using the informed measure, prunedMeasure_ will be set to si_->getSpaceMeasure();
  r_rrg_ = rewireFactor_ * 2.0 * std::pow((1.0 + 1.0/dimDbl) * (prunedMeasure_ / ompl::unitNBallMeasure(si_->getStateDimension())), 1.0 / dimDbl);
}

} // namespace planning

} // namespace intel_wind
