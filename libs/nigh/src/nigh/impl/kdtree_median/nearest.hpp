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
#ifndef NIGH_IMPL_KDTREE_MEDIAN_NEAREST_HPP
#define NIGH_IMPL_KDTREE_MEDIAN_NEAREST_HPP

#include "strategy.hpp"
#include "node.hpp"
#include "traversal.hpp"
#include "traversals.hpp"
#include <cassert>

namespace unc::robotics::nigh::impl::kdtree_median {
    template <class Tree, class NearSet>
    class Nearest;

    template <
        class T,
        class Space,
        class KeyFn,
        class Concurrency,
        std::size_t minTreeSize,
        std::size_t linearSearchSize,
        class Allocator,
        class NearSet>
    class Nearest<
        Nigh<T, Space, KeyFn, Concurrency, KDTreeMedian<minTreeSize, linearSearchSize>, Allocator>,
        NearSet>
        : public NearSet
    {
        using Tree = Nigh<T, Space, KeyFn, Concurrency, KDTreeMedian<minTreeSize, linearSearchSize>, Allocator>;
        using Key = typename Space::Type;

        const Tree& tree_;
        const Key key_;

        Traversal<Tree, Key, typename Space::Metric> traversal_;
                
    public:
        template <typename K, typename ... Args>
        Nearest(const Tree& tree, K&& key, Args&& ... args)
            : NearSet(std::forward<Args>(args)...)
            , tree_(tree)
            , key_(std::forward<K>(key))
            , traversal_(tree.metricSpace())
        {
        }

        template <class Iter>
        void insert(Iter first, Iter last) {
            NearSet::insert(
                first, last,
                [&] (const T& t) { return tree_.distToKey(t, key_); });
        }
        

        template <class Iter>
        void operator() (const Node* node, Iter first, Iter last) {
            if (NearSet::dist() < traversal_.distToRegion())
                return;
            if (std::distance(first, last) <= linearSearchSize) {
                insert(first, last);
            } else {
                assert(node != nullptr);
                traversal_.follow(*this, tree_.metricSpace(),
                                  node, node->axis(), key_, first, last);
            }
        }
    };
}

#endif
