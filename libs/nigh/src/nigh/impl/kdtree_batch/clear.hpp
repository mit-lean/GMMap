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
#ifndef NIGH_IMPL_KDTREE_BATCH_CLEAR_HPP
#define NIGH_IMPL_KDTREE_BATCH_CLEAR_HPP

#include "types.hpp"
#include "traversal.hpp"

namespace unc::robotics::nigh::impl::kdtree_batch {

    template <typename Tree>
    class Clear {
        using Key = key_t<Tree>;
        using Node = node_t<Tree>;
        using Leaf = leaf_t<Tree>;

        Tree& tree_;
        Traversal<Tree, Key> traversal_;

    public:
        Clear(Tree& tree)
            : tree_(tree)
            , traversal_(tree.metricSpace())
        {
        }

        template <typename T>
        void dealloc(T* ptr) {
            tree_.dealloc(ptr);
        }

        void operator() (Node *node) {
            if (!node->isLeaf()) {
                traversal_.clear(*this, tree_.metricSpace(), node, node->axis());
            } else {
                Leaf *leaf = static_cast<Leaf*>(node);
                dealloc(leaf);
            }
        }
    };
}

#endif
