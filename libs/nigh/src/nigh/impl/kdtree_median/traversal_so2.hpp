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
#ifndef NIGH_IMPL_KDTREE_MEDIAN_TRAVERSAL_SO2_HPP
#define NIGH_IMPL_KDTREE_MEDIAN_TRAVERSAL_SO2_HPP

#include "traversal.hpp"
#include "lp_branch.hpp"
#include <Eigen/Dense>

namespace unc::robotics::nigh::impl::kdtree_median {
    template <class Tree, class Key, int p, class Get>
    class Traversal<Tree, Key, metric::SO2<p>, Get> {
        using Metric = metric::SO2<p>;
        using Space = metric::Space<Key, Metric>;
        using Distance = typename Space::Distance;

        static constexpr int dimensions = Space::dynamic
            ? Eigen::Dynamic : Space::kDimensions;

        Eigen::Array<Distance, dimensions, 2> bounds_;
        Eigen::Matrix<Distance, dimensions, 1> dists_;
        Distance distToRegion_{0};
        
    public:
        Traversal(const Space& space)
            : bounds_(space.dimensions(), 2)
            , dists_(space.dimensions())
        {
            bounds_.col(0) = -PI<Distance>;
            bounds_.col(1) = PI<Distance>;
            dists_.setZero();
        }

        Distance distToRegion() const {
            return distToRegion_;
        }

        template <typename Fn>
        void recur(const Key& key, unsigned a, int side, Distance split, const Fn& fn) {
            Distance b0 = bounds_(a, 0);
        }

        template <typename Nearest, typename Iter>
        void recur(
            Nearest& nearest,
            Distance split, Node *child, int childNo,
            unsigned axis,
            const Key& key,
            Iter first, Iter last)
        {
            Distance b0 = bounds_(axis, 0);
            Distance b1 = bounds_(axis, 1);

            // setting the antipodal bounds only needs to happen at
            // the top split in the tree.  We can potentially simplify
            // this code.
            if (b0 == -PI<Distance> && b1 == PI<Distance>)
                bounds_(axis, !childNo) = so2::antipode(split);

            bounds_(axis, childNo) = split;

            Distance d0 = so2::ccwDist(Space::coeff(key, axis), bounds_(axis, 0));
            Distance d1 = so2::ccwDist(bounds_(axis, 1), Space::coeff(key, axis));
            dists_[axis] = (d0 + d1 < 2*PI<Distance>) ? std::min(d0, d1) : 0;
            distToRegion_ = dists_.template lpNorm<p>();

            nearest(child, first, last);
            
            bounds_(axis, 0) = b0;
            bounds_(axis, 1) = b1;
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
                first, first + std::distance(first, last)/2, last}};
            
            Distance dist = so2::ccwDist(branch->split(), Space::coeff(key, axis));
            int childNo = dist > PI<Distance>;
            
            Distance oldDist = dists_[axis];
            Distance oldDistToRegion = distToRegion_;

            recur(nearest, branch->split(), branch->child(childNo), childNo,
                  axis, key, iters[childNo], iters[childNo+1]);
            recur(nearest, branch->split(), branch->child(childNo), !childNo,
                  axis, key, iters[1-childNo], iters[2-childNo]);

            dists_[axis] = oldDist;
            distToRegion_ = oldDistToRegion;
        }
    };
}

#endif
