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

// This software has been modified from the original implementation
// with the above copyright and license.  Additional portions of this
// software are covered by the following license agreement.
//
// Software License Agreement (BSD-3-Clause)
//
// Copyright 2018 The University of North Carolina at Chapel Hill
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

//! @author Mark Moll, Bryant Gipson, Jeff Ichnowski

#pragma once

#ifndef NIGH_GNAT_HPP
#define NIGH_GNAT_HPP

#include "nigh_forward.hpp"
#include "metric/space.hpp"
#include "impl/nearest_base.hpp"
#include "impl/near_set.hpp"
#include "impl/compare_nth.hpp"
#include "impl/k_centers.hpp"
#include "impl/locked_nearest.hpp"
#include <queue>
#include <random>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

namespace unc::robotics::nigh {

    namespace impl {
        template <typename T>
        class RemovedCache {
            std::set<T> set_;

        public:
            bool empty() const { return set_.empty(); }
        };
    }

    template <
        unsigned degree = 8, unsigned minDegree = 4, unsigned maxDegree = 12,
        unsigned maxNumPtsPerLeaf = 50, unsigned removedCacheSize = 500,
        bool rebalancing = false>
    struct GNAT {};

    template <
        typename T,
        typename Space,
        typename KeyFn,
        typename Concurrency,
        unsigned degree, unsigned minDegree, unsigned maxDegree,
        unsigned maxNumPtsPerLeaf, unsigned removedCacheSize,
        bool rebalancing,
        typename Allocator>
    class Nigh<
        T, Space, KeyFn, Concurrency,
        GNAT<degree, minDegree, maxDegree,
             maxNumPtsPerLeaf, removedCacheSize,
             rebalancing>,
        Allocator>
        : public impl::NearestBase<
            Nigh<
                T, Space, KeyFn, Concurrency,
                GNAT<degree, minDegree, maxDegree,
                    maxNumPtsPerLeaf, removedCacheSize,
                    rebalancing>,
                Allocator>>
    {
        static_assert(
            std::is_same_v<Concurrency, NoThreadSafety> ||
            std::is_same_v<Concurrency, ConcurrentRead>,
            "invalid type of concurrency.  must be Concurrent, ConcurrentRead, NoThreadSafety");

        using Base =
            impl::NearestBase<
            Nigh<
                T, Space, KeyFn, Concurrency,
                GNAT<degree, minDegree, maxDegree,
                    maxNumPtsPerLeaf, removedCacheSize,
                    rebalancing>,
                Allocator>>;

    public:
        using Key = typename Space::Type;
        using Distance = typename Space::Distance;

    private:
        using RNG = std::conditional_t<sizeof(Distance) == 4, std::mt19937, std::mt19937_64>;

        struct Node;
        using NodeDist = std::pair<Node*, Distance>;

        struct NodeDistCompare {
            bool operator()(const NodeDist& a, const NodeDist& b) const {
                return (a.second - a.first->maxRadius_) > (b.second - b.first->maxRadius_);
            }
        };

        using CompareSecond = impl::CompareNth<1>;

        using NodeQueue = std::priority_queue<NodeDist, std::vector<NodeDist>, NodeDistCompare>;

        struct Node {
            unsigned degree_;
            T pivot_;

            Distance minRadius_{ std::numeric_limits<Distance>::infinity()};
            Distance maxRadius_{-std::numeric_limits<Distance>::infinity()};

            std::vector<Distance> minRange_;
            std::vector<Distance> maxRange_;
            std::vector<T> data_;
            std::vector<Node*> children_;

            Node(unsigned deg, const T& pivot)
                : degree_(deg)
                , pivot_(pivot)
                , minRange_(deg, std::numeric_limits<Distance>::infinity())
                , maxRange_(deg,-std::numeric_limits<Distance>::infinity())
            {
                data_.reserve(maxNumPtsPerLeaf+1);
            }

            ~Node() {
                for (unsigned i=0 ; i<children_.size() ; ++i)
                    delete children_[i];
            }

            void updateRadius(Distance d) {
                if (minRadius_ > d)
                    minRadius_ = d;
                if (maxRadius_ < d)
                    maxRadius_ = d;
            }

            void updateRange(unsigned i, Distance d) {
                if (minRange_[i] > d)
                    minRange_[i] = d;
                if (maxRange_[i] < d)
                    maxRange_[i] = d;
            }

            void add(Nigh& gnat, const T& data) {
                const Key& key = gnat.getKey(data);
                if (children_.empty()) {
                    data_.push_back(data);
                    if (needToSplit(gnat)) {
                        if (!gnat.removed_.empty()) {
                            gnat.rebuildDataStructure();
                        } else if (gnat.size_ >= gnat.rebuildSize_) {
                            gnat.rebuildSize_ <<= 1;
                            gnat.rebuildDataStructure();
                        } else {
                            split(gnat);
                        }
                    }
                } else {
                    std::vector<Distance> dist(children_.size());
                    Distance minDist = dist[0] = gnat.distance(key, gnat.getKey(children_[0]->pivot_));
                    int minInd = 0;

                    for (unsigned i=1 ; i<children_.size() ; ++i)
                        if ((dist[i] = gnat.distance(key, gnat.getKey(children_[i]->pivot_))) < minDist)
                            minDist = dist[minInd = i];

                    for (unsigned i=0 ; i<children_.size() ; ++i)
                        children_[i]->updateRange(minInd, dist[i]);

                    children_[minInd]->updateRadius(minDist);
                    children_[minInd]->add(gnat, data);
                }
            }

            bool needToSplit(const Nigh& gnat) const {
                std::size_t n = data_.size();
                return n > maxNumPtsPerLeaf && n > degree_;
            }

            void split(Nigh& gnat) {
                impl::KCenters<T, Distance, degree, maxNumPtsPerLeaf+1> kCenters;
                kCenters.compute(
                    data_, degree_, gnat.rng_,
                    [&] (const T& a, const T& b) {
                        return gnat.distance(gnat.getKey(a), gnat.getKey(b));
                    });

                children_.reserve(degree_);
                unsigned numCenters = kCenters.numCenters();
                for (unsigned i=0 ; i<numCenters ; ++i)
                    children_.push_back(new Node(degree_, data_[kCenters.center(i)]));

                degree_ = kCenters.numCenters();
                for (unsigned j=0 ; j<data_.size() ; ++j) {
                    unsigned k = 0;
                    for (unsigned i=1 ; i<degree_ ; ++i)
                        if (kCenters.dist(j, i) < kCenters.dist(j, k))
                            k = i;
                    Node *child = children_[k];
                    if (j != kCenters.center(k)) {
                        child->data_.push_back(data_[j]);
                        child->updateRadius(kCenters.dist(j, k));
                    }
                    for (unsigned i=0 ; i<degree_ ; ++i)
                        children_[i]->updateRange(k, kCenters.dist(j, i));
                }

                for (unsigned i=0 ; i<degree_ ; ++i) {
                    children_[i]->degree_ =
                        std::min(std::max((unsigned)((degree_ * children_[i]->data_.size()) / data_.size()),
                                          minDegree),
                                 maxDegree);
                    if (children_[i]->minRadius_ >= std::numeric_limits<Distance>::infinity())
                        children_[i]->minRadius_ = children_[i]->maxRadius_ = 0;
                }

                // use swap() to clear() and revert to default capacity.
                std::vector<T> tmp;
                data_.swap(tmp);
                for (unsigned i=0 ; i<degree_ ; ++i)
                    if (children_[i]->needToSplit(gnat))
                        children_[i]->split(gnat);
            }

            template <typename NearSet>
            void nearestK(
                const Nigh& gnat, const Key& key,
                NearSet& nearSet,
                NodeQueue& nodeQueue,
                bool &isPivot) const
            {
                for (unsigned i=0 ; i<data_.size() ; ++i)
                    if (!gnat.isRemoved(data_[i])) {
                        if (nearSet.insert(data_[i], gnat.distance(key, gnat.getKey(data_[i]))))
                            isPivot = false;
                    }

                if (children_.empty())
                    return;

                Distance dist;
                std::vector<Distance> distToPivot(children_.size());
                std::vector<int> permutation(children_.size());
                std::iota(permutation.begin(), permutation.end(), 0);
                std::shuffle(permutation.begin(), permutation.end(), gnat.rng_);

                for (unsigned i=0 ; i<children_.size() ; ++i) {
                    if (permutation[i] >= 0) {
                        Node *child = children_[permutation[i]];
                        distToPivot[permutation[i]] = gnat.distance(key, gnat.getKey(child->pivot_));
                        if (nearSet.insert(child->pivot_, distToPivot[permutation[i]]))
                            isPivot = true;
                        if (nearSet.full()) {
                            dist = nearSet.dist();
                            for (unsigned j = 0 ; j<children_.size() ; ++j)
                                if (permutation[j] >= 0 && i != j &&
                                    (distToPivot[permutation[i]] - dist > child->maxRange_[permutation[j]] ||
                                     distToPivot[permutation[i]] + dist < child->minRange_[permutation[j]]))
                                    permutation[j] = -1;
                        }
                    }
                }

                dist = nearSet.dist();
                for (unsigned i=0 ; i<children_.size() ; ++i) {
                    if (permutation[i] >= 0) {
                        Node *child = children_[permutation[i]];
                        if (!nearSet.full() || (distToPivot[permutation[i]] - dist <= child->maxRadius_ &&
                                                distToPivot[permutation[i]] + dist >= child->minRadius_)) {
                            nodeQueue.emplace(child, distToPivot[permutation[i]]);
                        }
                    }
                }
            }

            void list(const Nigh& gnat, std::vector<T>& data) const {
                if (!gnat.isRemoved(pivot_))
                    data.push_back(pivot_);
                for (const auto& t : data_)
                    if (!gnat.isRemoved(t))
                        data.push_back(t);
                for (const Node* c : children_)
                    c->list(gnat, data);
            }

            template <typename Fn>
            void visit(const Nigh& gnat, const Fn& fn) const {
                if (!gnat.isRemoved(pivot_))
                    fn(pivot_);
                for (const auto& t : data_)
                    if (!gnat.isRemoved(t))
                        fn(t);
                for (const Node* c : children_)
                    c->visit(gnat, fn);
            }
        };

        mutable RNG rng_;
        Node *tree_{nullptr};
        std::size_t size_{0};
        std::size_t rebuildSize_{rebalancing ? maxNumPtsPerLeaf * degree : std::numeric_limits<std::size_t>::max()};
        impl::RemovedCache<T> removed_;

        bool isRemoved(const T& data) const {
            // TODO: return !removed_.empty() && removed_.find(data) != removed_.end();
            return false;
        }

        template <typename NearSet>
        bool nearestKInternal(const Key& key, NearSet& nearSet) const {
            bool isPivot;
            Distance dist;
            NodeDist nodeDist;
            NodeQueue nodeQueue_;

            dist = Base::distToKey(tree_->pivot_, key);
            isPivot = nearSet.insert(tree_->pivot_, dist);
            tree_->nearestK(*this, key, nearSet, nodeQueue_, isPivot);
            while (!nodeQueue_.empty()) {
                dist = nearSet.dist();
                nodeDist = nodeQueue_.top();
                nodeQueue_.pop();
                if (nearSet.full() && (nodeDist.second > nodeDist.first->maxRadius_ + dist ||
                                       nodeDist.second < nodeDist.first->minRadius_ - dist))
                    continue;
                nodeDist.first->nearestK(*this, key, nearSet, nodeQueue_, isPivot);
            }

            return isPivot;
        }

    public:
        Nigh(Nigh&& other)
            : Base(std::move(other))
            , rng_(std::move(other.rng_))
            , tree_(std::exchange(other.tree_, nullptr))
            , size_(std::exchange(other.size_, 0))
            , rebuildSize_(other.rebuildSize_)
            , removed_(std::move(other.removed_))
        {
        }

        Nigh(const Space& space = Space(), const KeyFn& getKey = KeyFn(), const Allocator& alloc = Allocator())
            : Base(space, getKey, alloc)
        {
        }

        ~Nigh() {
            clear();
        }

        void insert(const T& data) {
            if (tree_) {
                if (isRemoved(data))
                    rebuildDataStructure();
                tree_->add(*this, data);
                ++size_;
            } else {
                tree_ = new Node(degree, data);
                size_ = 1;
            }
        }

        template <typename Iter>
        void insert(Iter begin, Iter end) {
            if (begin == end)
                return;

            if (tree_) {
                for (Iter it = begin ; it != end ; ++it)
                    insert(*it);
            } else {
                tree_ = new Node(degree, *begin);
                for (Iter it = begin ; ++it != end ; )
                    tree_->data_.push_back(*it);
                size_ += std::distance(begin, end);
                if (tree_->needToSplit(*this))
                    tree_->split(*this);
            }
        }

        void rebuildDataStructure() {
            std::vector<T> all;
            list(all);
            clear();
            insert(all.begin(), all.end());
        }

        void clear() {
            if (tree_) {
                delete tree_;
                tree_ = nullptr;
            }
            size_ = 0;
            // TODO: removed_.clear();
            if (rebuildSize_ != std::numeric_limits<std::size_t>::max())
                rebuildSize_ = maxNumPtsPerLeaf * degree;
        }

        bool remove(const T& data) {
            //TODO
            // if (size_ == 0)
            //     return false;

            // std::vector<std::pair<T, Distance>> nbh;
            // bool isPivot = nearestKInternal(data, 1, nbh);
            // const T& d = nbh.front().first;
            // if (d != data)
            //     return false;
            // removed_.insert(d);
            // --size_;
            // if (isPivot || removed_.size() >= removedCacheSize)
            //     rebuildDataStructure();

            // return true;
            return false;
        }

        std::optional<std::pair<T, Distance>> nearest(const Key& key) const {
            impl::Near1Set<T, Distance> nearSet;
            if (tree_)
                nearestKInternal(key, nearSet);
            return nearSet.result();
        }

        template <typename Tuple, typename K, typename ResultAllocator>
        void nearest(
            std::vector<Tuple, ResultAllocator>& nbh,
            const K& key,
            std::size_t k,
            Distance maxRadius = std::numeric_limits<Distance>::infinity()) const
        {
            impl::NearKSet<Tuple, Distance, ResultAllocator> nearSet(nbh, k, maxRadius);
            if (tree_)
                nearestKInternal(key, nearSet);
            nearSet.sort();
        }

        std::size_t size() const {
            return size_;
        }

        template <typename Fn>
        void visit(const Fn& fn) const {
            if (tree_)
                tree_->visit(*this, fn);
        }

        void list(std::vector<T>& data) const {
            data.clear();
            data.reserve(size_);
            visit([&] (const T& t) { data.push_back(t); });
        }

        std::vector<T> list() const {
            std::vector<T> result;
            list(result);
            return result;
        }
    };

    // Specialization for R/W concurrency.  This version is a
    // read/write locked version of ConcurrentRead.
    template <
        typename T,
        typename Space,
        typename KeyFn,
        unsigned degree, unsigned minDegree, unsigned maxDegree,
        unsigned maxNumPtsPerLeaf, unsigned removedCacheSize,
        bool rebalancing,
        typename Allocator>
    class Nigh<
        T, Space, KeyFn, Concurrent,
        GNAT<degree, minDegree, maxDegree, maxNumPtsPerLeaf, removedCacheSize, rebalancing>,
        Allocator>
        : public impl::LockedNearest<
            Nigh<
                T, Space, KeyFn, ConcurrentRead,
                 GNAT<degree, minDegree, maxDegree, maxNumPtsPerLeaf, removedCacheSize, rebalancing>,
                 Allocator>>
    {
        using Base = impl::LockedNearest<
            Nigh<
                T, Space, KeyFn, ConcurrentRead,
                GNAT<degree, minDegree, maxDegree, maxNumPtsPerLeaf, removedCacheSize, rebalancing>,
                Allocator>>;

    public:
        using Base::Base;
    };
}

#endif // NIGH_GNAT_HPP
