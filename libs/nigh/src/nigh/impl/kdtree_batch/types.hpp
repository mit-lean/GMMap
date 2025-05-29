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
#ifndef NIGH_IMPL_KDTREE_BATCH_TYPES_HPP
#define NIGH_IMPL_KDTREE_BATCH_TYPES_HPP

#include "../region.hpp"
#include "../regions.hpp"
#include "../../nigh_forward.hpp"
#include "strategy.hpp"
#include "node.hpp"
#include "leaf.hpp"

namespace unc::robotics::nigh::impl::kdtree_batch {
    template <typename Tree>
    struct Types;

    template <
        typename T,
        typename Space_,
        typename KeyFn_,
        typename Concurrency_,
        std::size_t batchSize,
        typename Allocator_>
    struct Types<Nigh<T, Space_, KeyFn_, Concurrency_, KDTreeBatch<batchSize>, Allocator_>> {

        using Value = T;
        using KeyFn = KeyFn_;
        using Space = Space_;
        using Key = typename Space::Type;
        using Concurrency = Concurrency_;
        using Metric = typename Space::Metric;
        using Distance = typename Space::Distance;

        using Node = kdtree_batch::Node<T, Space, Concurrency>;
        using Leaf = kdtree_batch::Leaf<T, Space, Concurrency, batchSize>;

        static constexpr bool kConcurrentWrites = std::is_same_v<Concurrency, Concurrent>;

        using NodePointer = Atom<Node*, kConcurrentWrites>;

        static constexpr std::size_t kBatchSize = batchSize;
    };


    template <typename Tree>
    using value_t = typename Types<Tree>::Value;
    template <typename Tree>
    using distance_t = typename Types<Tree>::Distance;
    template <typename Tree>
    using concurrency_t = typename Types<Tree>::Concurrency;
    template <typename Tree>
    using key_t = typename Types<Tree>::Key;
    template <typename Tree>
    using space_t = typename Types<Tree>::Space;
    template <typename Tree>
    using keyfn_t = typename Types<Tree>::KeyFn;
    template <typename Tree>
    using node_t = typename Types<Tree>::Node;
    template <typename Tree>
    using leaf_t = typename Types<Tree>::Leaf;
    template <typename Tree>
    using node_pointer_t = typename Types<Tree>::NodePointer;

    template <typename Tree>
    static constexpr std::size_t kBatchSize = Types<Tree>::kBatchSize;
}

#endif
