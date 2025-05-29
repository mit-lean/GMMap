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
#ifndef NIGH_IMPL_KDTREE_MEDIAN_TRAVERSAL_LP_HPP
#define NIGH_IMPL_KDTREE_MEDIAN_TRAVERSAL_LP_HPP

#include "traversal.hpp"
#include "lp_branch.hpp"

namespace unc::robotics::nigh::impl::kdtree_median {
    template <class Tree, class Key, int p, class Get>
    class Traversal<Tree, Key, metric::LP<p>, Get> {
        using Metric = metric::LP<p>;
        using Space = metric::Space<Key, Metric>;
        using Distance = typename Space::Distance;

        Eigen::Matrix<Distance, Space::kDimensions, 1> deltas_;
        Distance distToRegion_{0};
        
    public:
        Traversal(const Space& space)
            : deltas_(space.dimensions())
        {
            deltas_.setZero();
        }

        Distance distToRegion() const {
            return distToRegion_;
        }

        template <typename Nearest, typename Iter>
        void follow(
            Nearest& nearest,
            const Space&,
            const Node *node, unsigned axis,
            const Key& key,
            Iter first, Iter last)
        {
            const auto* branch = static_cast<const LPBranch<Distance>*>(node);
            std::array<Iter, 3> iters{{
                    first,
                    first + std::distance(first, last)/2,
                    last}};
            Distance delta = Space::coeff(key, axis) - branch->split();
            int childNo = delta > 0;
            
            nearest(branch->child(childNo), iters[childNo], iters[childNo+1]);

            Distance oldDelta = deltas_[axis];
            deltas_[axis] = delta;
            Distance oldDist = distToRegion_;
            distToRegion_ = deltas_.template lpNorm<p>();

            nearest(branch->child(!childNo), iters[1-childNo], iters[2-childNo]);
            
            deltas_[axis] = oldDelta;
            distToRegion_ = oldDist;
        }
    };
}

#endif
