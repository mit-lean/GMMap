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
#ifndef NIGH_IMPL_KDTREE_BATCH_NEAREST_TRAVERSAL_SO2_HPP
#define NIGH_IMPL_KDTREE_BATCH_NEAREST_TRAVERSAL_SO2_HPP

#include "nearest_traversal.hpp"
#include "traversal_so2.hpp"
#include <Eigen/Dense>

namespace unc::robotics::nigh::impl::kdtree_batch {
    template <typename Tree, typename Key, int p, typename Get>
    class NearestTraversal<Tree, Key, metric::SO2<p>, Get>
        : public Traversal<Tree, Key, metric::SO2<p>, Get>
    {
        using Base = Traversal<Tree, Key, metric::SO2<p>, Get>;
        using Distance = typename Base::Distance;
        using SO2Branch = typename Base::SO2Branch;
        using Node = typename Base::Node;
        using Concurrency = typename Base::Concurrency;
        using Space = typename Base::Space;

        static constexpr int dimensions = Space::dynamic
            ? Eigen::Dynamic : Space::kDimensions;

        Distance distToRegion_{0};
        Eigen::Array<Distance, dimensions, 2> bounds_;
        Eigen::Matrix<Distance, dimensions, 1> dists_;

    public:
        NearestTraversal(const Space& space)
            : Base(space)
        {
            bounds_.col(0) = -PI<Distance>;
            bounds_.col(1) = PI<Distance>;
            dists_.setZero();
        }

        constexpr Distance distToRegion(const Key&, const Region<Key, metric::SO2<p>, Concurrency>&) const {
            return distToRegion_;
        }

        template <typename Fn>
        void recur(const Key& key, unsigned a, int side, Distance split, const Fn& fn) {
            //
            //      @@@|@@@
            //     @@@@|@@@@
            // pi -----+----- 0
            //     @@@@|@@@@
            //      @@@|@@@
            //

            Distance b0 = bounds_(a, 0);
            Distance b1 = bounds_(a, 1);
            Distance oldDist = dists_[a];
            Distance oldDistToRegion = distToRegion_;
            assert(so2::ccwDist(b0, split) < so2::ccwRange(b0, b1));

            // TODO: we only have to do this the first time, we
            // can simplify this check.

            // check if antipode splits cuts through other side
            if (bounds_(a, 0) == -PI<Distance> && bounds_(a, 1) == PI<Distance>)
                bounds_(a, !side) = so2::antipode(split);

            bounds_(a, side) = split;

            Distance d0 = so2::ccwDist(Space::coeff(key, a), bounds_(a, 0));
            Distance d1 = so2::ccwDist(bounds_(a, 1), Space::coeff(key, a));
            dists_[a] = (d0+d1 < 2*PI<Distance>) ? std::min(d0, d1) : 0;
            distToRegion_ = dists_.template lpNorm<p>();

            fn();

            bounds_(a, 0) = b0;
            bounds_(a, 1) = b1;
            dists_[a] = oldDist;
            distToRegion_ = oldDistToRegion;
        }

        template <typename Visitor>
        void traverse(Visitor& visitor, const Space&, const Node *node, unsigned axis, const Key& key) {
            const SO2Branch *branch = static_cast<const SO2Branch*>(node);
            Distance dist = so2::ccwDist(branch->split(), Space::coeff(key, axis));
            int side = dist > PI<Distance>;

            //Distance oldDist = distToRegion_;
            recur(key, axis, side, branch->split(), [&] {
                    // assert(oldDist == distToRegion_);
                    visitor(branch->child(side));
                });
            recur(key, axis,!side, branch->split(), [&] {
                    visitor(branch->child(!side));
                });
        }
    };
}

#endif
