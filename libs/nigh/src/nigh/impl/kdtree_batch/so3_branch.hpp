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
#ifndef NIGH_IMPL_KDTREE_BATCH_SO3_BRANCH_HPP
#define NIGH_IMPL_KDTREE_BATCH_SO3_BRANCH_HPP

#include "branch.hpp"

namespace unc::robotics::nigh::impl::kdtree_batch {
    template <typename T, typename Space, typename Concurrency, std::size_t batchSize>
    class SO3Branch : public Branch<T, Space, Concurrency, batchSize> {
        using Base = Branch<T, Space, Concurrency, batchSize>;
        using Node = kdtree_batch::Node<T, Space, Concurrency>;
        using Leaf = kdtree_batch::Leaf<T, Space, Concurrency, batchSize>;
        using Distance = typename Space::Distance;
        using Split = Eigen::Matrix<Distance, 2, 1>;

        static constexpr bool concurrentWrites = std::is_same_v<Concurrency, Concurrent>;

        Split split_;
        std::array<Atom<Node*, concurrentWrites>, 2> children_;

    public:
        SO3Branch(Leaf *leaf, int axis, const Split& split, Leaf *c0, Leaf *c1)
            : Base(leaf, axis)
            , split_(split)
            , children_{{c0, c1}}
        {
        }

        const Split& split() const { return split_; }
        auto& child(int i) { return children_[i]; }
        const auto& child(int i) const { return children_[i]; }
    };
}

#endif
