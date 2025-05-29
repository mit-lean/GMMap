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
#ifndef NIGH_IMPL_KDTREE_BATCH_NODE_HPP
#define NIGH_IMPL_KDTREE_BATCH_NODE_HPP

#include "../region.hpp"
#include "../regions.hpp"

namespace unc::robotics::nigh::impl::kdtree_batch {
    template <std::size_t I, std::size_t N, typename Tree, typename Key, typename Metric, typename Get>
    struct CartesianHelper;

    template <typename T, typename Space, typename Concurrency>
    class Node : Region<typename Space::Type, typename Space::Metric, Concurrency> {
    protected:
        using Key = typename Space::Type;
        using Metric = typename Space::Metric;
        using NodeRegion = Region<Key, Metric, Concurrency>;

        int axis_;

        // the cartesian helper needs to modify axis_;
        template <std::size_t, std::size_t, typename, typename, typename, typename>
        friend struct CartesianHelper;

    public:
        static constexpr int kLeafAxis = -1;

        // Leaf
        template <typename Traversal>
        Node(const Space& space, const Traversal& traversal, const Key& q)
            : NodeRegion(space, traversal, q)
            , axis_(kLeafAxis)
        {
        }

        // Branch
        Node(const NodeRegion& region, int axis)
            : NodeRegion(region)
            , axis_(axis)
        {
            assert(axis >= 0);
        }

        constexpr int axis() const { return axis_; }
        constexpr bool isLeaf() const { return axis_ == kLeafAxis; }
        constexpr NodeRegion& region() { return *this; }
        constexpr const NodeRegion& region() const { return *this; }
    };
}

#endif
