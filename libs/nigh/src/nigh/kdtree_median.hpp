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

//! @author Jeff Ichnowski

#pragma once
#ifndef NIGH_KDTREE_MEDIAN_HPP
#define NIGH_KDTREE_MEDIAN_HPP

#include "nigh_forward.hpp"
#include "metric/space.hpp"
#include "impl/nearest_base.hpp"
#include "impl/atom.hpp"
#include "impl/bits.hpp"
#include "impl/near_set.hpp"
#include "impl/block_allocator.hpp"
#include "impl/locked_nearest.hpp"
#include "impl/kdtree_median/node.hpp"
#include "impl/kdtree_median/nearest.hpp"
#include "impl/kdtree_median/builder.hpp"
#include <cassert>
#include <utility>
#include <iostream> // TODO: REMOVE!

namespace unc::robotics::nigh {

    template <
        typename T,
        typename Space,
        typename KeyFn,
        typename Concurrency,
        std::size_t minTreeSize,
        std::size_t linearSearchSize,
        typename Allocator>
    class Nigh<T, Space, KeyFn, Concurrency, KDTreeMedian<minTreeSize, linearSearchSize>, Allocator>
        : public impl::NearestBase<
            Nigh<T, Space, KeyFn, Concurrency, KDTreeMedian<minTreeSize, linearSearchSize>, Allocator>>
    {
        using Base = impl::NearestBase<Nigh>;
    public:
        using Key = typename Base::Key;
        using Metric = typename Base::Metric;
        using Distance = typename Base::Distance;

    private:
        using Blocks = impl::BlockAllocator<0, Allocator>;
        using Marker = typename Blocks::Marker;
        using Builder = impl::kdtree_median::Builder<Nigh>;
        using Node = impl::kdtree_median::Node;
        using Root = std::pair<Node*, Marker>;
        using Values = std::vector<T, Allocator>;
        using ValueIter = typename Values::iterator;
        
        static constexpr std::size_t minTreeMask = ~(minTreeSize - 1);
        
        Blocks blocks_;
        std::vector<Root> roots_;
        Values values_;

        friend class impl::kdtree_median::Builder<Nigh>;
        template <class Tree, class Set>
        friend class impl::kdtree_median::Nearest;
        
        void addOne() {
            std::size_t s = values_.size();
            std::size_t newTreeSize = ((s^(s-1)) + 1) >> 1;

            // std::clog << "addOne: " << s << " : " << newTreeSize << std::endl;
            
            if (newTreeSize >= minTreeSize) {
                std::size_t numTrees = impl::popcount(s & minTreeMask);
                assert(numTrees > 0);
                
                if (roots_.size() >= numTrees) {
                    blocks_.rollback(roots_[numTrees-1].second);
                    roots_.resize(numTrees-1);
                }

                // std::clog << "building tree " << newTreeSize << std::endl;

                // Builder builder{Base::metricSpace(), Base::keyFn(), blocks_}
                Builder builder(*this);
                auto mark = blocks_.mark();
                Node *node = builder(values_.end() - newTreeSize, values_.end());
                roots_.emplace_back(node, mark);
            }
        }

        template <class Nearest>
        void scan(Nearest& nearest) const {
            auto rootIt = roots_.begin();
            auto elemIt = values_.begin();
            for (std::size_t remaining = size() ; remaining >= minTreeSize ; ++rootIt) {
                std::size_t treeSize = 1 << impl::log2(remaining);
                nearest(rootIt->first, elemIt, elemIt + treeSize);
                elemIt += treeSize;
                remaining &= ~treeSize;
            }

            assert(rootIt == roots_.end());
            nearest.insert(elemIt, values_.end());
        }
        
    public:
        Nigh(const Nigh&) = delete;
        Nigh(Nigh&& other)
            : Base(std::move(other))
            , blocks_(std::move(other.blocks_))
            , roots_(std::move(other.roots_))
            , values_(std::move(other.values_))
        {
        }

        explicit Nigh(
            const Space& metric = Space(),
            const KeyFn& member = KeyFn(),
            Allocator allocator = Allocator())
            : Base(metric, member, allocator)
        {
        }

        std::size_t size() const {
            return values_.size();
        }

        void insert(const T& value) {
            // std::clog << "insert" << std::endl;
            values_.push_back(value);
            addOne();
        }

        template <typename K>
        std::optional<std::pair<T, Distance>> nearest(const K& q) const {
            impl::kdtree_median::Nearest<Nigh, impl::Near1Set<T, Distance>> nearest(*this, q);
            scan(nearest);
            return nearest.result();
        }

        template <typename K>
        std::optional<T> nearest(const K& q, Distance* dist) const {
            impl::kdtree_median::Nearest<Nigh, impl::Near1Set<T, Distance>> nearest(*this, q);
            scan(nearest);
            return nearest.result(dist);
        }

        template <typename Tuple, typename K, typename ResultAllocator>
        void nearest(
            std::vector<Tuple, ResultAllocator>& nbh,
            const K& q,
            std::size_t k,
            Distance maxRadius = std::numeric_limits<Distance>::infinity()) const
        {
            impl::kdtree_median::Nearest<Nigh, impl::NearKSet<Tuple, Distance, ResultAllocator>>
                nearest(*this, q, nbh, k, maxRadius);
            
            scan(nearest);
            nearest.sort();
        }

        std::vector<T> list() const {
            return values_; // this is a copy!
        }
    };

    // Specialization for R/W concurrency.  This version is a
    // read/write locked version of ConcurrentRead
    template <
        typename T,
        typename Space,
        typename KeyFn,
        std::size_t minTreeSize,
        typename Allocator>
    class Nigh<T, Space, KeyFn, Concurrent, KDTreeMedian<minTreeSize>, Allocator>
        : public impl::LockedNearest<
            Nigh<T, Space, KeyFn, ConcurrentRead, KDTreeMedian<minTreeSize>, Allocator>>
    {
        using Base = impl::LockedNearest<
            Nigh<T, Space, KeyFn, ConcurrentRead, KDTreeMedian<minTreeSize>, Allocator>>;
        
    public:
        using Base::Base;
    };
    
}

#endif
