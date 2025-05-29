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
#ifndef NIGH_IMPL_KDTREE_BATCH_NEAREST_HPP
#define NIGH_IMPL_KDTREE_BATCH_NEAREST_HPP

#include "node.hpp"
#include "nearest_traversal.hpp"
#include "nearest_traversals.hpp"

namespace unc::robotics::nigh::impl::kdtree_batch {

    template <typename Tree, typename NearSet>
    class Nearest;

    template <
        typename T,
        typename Space,
        typename KeyFn,
        typename Concurrency_,
        std::size_t batchSize,
        typename Allocator,
        typename NearSet>
    class Nearest<
        Nigh<T, Space, KeyFn, Concurrency_, KDTreeBatch<batchSize>, Allocator>,
        NearSet> : public NearSet
    {
        using Tree = Nigh<T, Space, KeyFn, Concurrency_, KDTreeBatch<batchSize>, Allocator>;
        using Key = typename Space::Type;
        using Node = kdtree_batch::Node<T, Space, Concurrency_>;
        using Leaf = kdtree_batch::Leaf<T, Space, Concurrency_, batchSize>;

        const Tree& tree_;
        NearestTraversal<Tree> traversal_;

        const Key key_;

    public:
        template <typename K, typename ... Args>
        Nearest(const Tree& tree, K&& key, Args&& ... args)
            : NearSet(std::forward<Args>(args)...)
            , tree_(tree)
            , traversal_(tree.metricSpace())
            , key_(std::forward<K>(key))
        {
        }

        __attribute__((always_inline))
        void operator() (const Node* node) {
            if (NearSet::dist() < traversal_.distToRegion(key_, node->region()))
                return;
            if (!node->isLeaf()) {
                traversal_.traverse(*this, tree_.metricSpace(), node, node->axis(), key_);
            } else {
                const Leaf *leaf = static_cast<const Leaf*>(node);
                int size = std::abs(leaf->size());
                NearSet::insert(
                    leaf->elements(), leaf->elements() + size,
                    [&] (const T& t) { return tree_.distToKey(t, key_); });
            }
        }

    };
}

#endif
