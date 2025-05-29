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
#ifndef NIGH_IMPL_KDTREE_BATCH_NEAREST_TRAVERSAL_SO3_HPP
#define NIGH_IMPL_KDTREE_BATCH_NEAREST_TRAVERSAL_SO3_HPP

#include "nearest_traversal.hpp"
#include "traversal_so3.hpp"
#include <cmath>

namespace unc::robotics::nigh::impl::kdtree_batch {
    template <typename Tree, typename Key, typename Get>
    class NearestTraversal<Tree, Key, metric::SO3, Get> : public Traversal<Tree, Key, metric::SO3, Get> {
        using Base = Traversal<Tree, Key, metric::SO3, Get>;
        using Node = typename Base::Node;
        using SO3Root = typename Base::SO3Root;
        using SO3Branch = typename Base::SO3Branch;
        using Concurrency = typename Base::Concurrency;
        using Space = typename Base::Space;
        using Distance = typename Space::Distance;

        Distance distToRegion_{0};
        Distance dotMax_{0};
        Eigen::Matrix<Distance, 2, 3> dots_;

    public:
        using Base::Base;

        constexpr Distance distToRegion(const Key& key, const Region<Key, metric::SO3, Concurrency>&) const {
            return distToRegion_;
        }

        void initVol(const Key& key) {
            Distance qv = Space::coeff(key, Base::vol_);
            Distance s = qv < 0 ? -SQRT1_2<Distance> : SQRT1_2<Distance>;
            dotMax_ = 0;
            for (unsigned a=0 ; a<3 ; ++a) {
                Distance qa = Space::coeff(key, (Base::vol_ + a + 1)%4);
                dots_(0, a) = (qa + qv) * s;
                dots_(1, a) = (qa - qv) * s;
                if (dots_(0, a) < 0 || dots_(1, a) > 0)
                    dotMax_ = std::max(dotMax_, dots_.col(a).cwiseAbs().minCoeff());
            }
            distToRegion_ = std::asin(dotMax_);
        }

        template <typename Visitor>
        // __attribute__((always_inline))
        void recur(Visitor& visitor, const SO3Branch *branch, unsigned axis, int childNo, Distance dot) {
            assert(axis < 3);
            assert(childNo == 0 || childNo == 1);

            Distance oldDist = distToRegion_;
            Distance oldDotMax = dotMax_;
            Distance oldDot = dots_(!childNo, axis);

            dots_(!childNo, axis) = dot;
            if (dots_(0, axis) < 0 || dots_(1, axis) > 0) {
                Distance minDot = dots_.col(axis).cwiseAbs().minCoeff();
                if (minDot > oldDotMax)
                    distToRegion_ = std::asin(dotMax_ = minDot);
            }

            visitor(branch->child(childNo));

            dots_(!childNo, axis) = oldDot;
            dotMax_ = oldDotMax;
            distToRegion_ = oldDist;
        }

        template <typename Visitor>
        void traverse(Visitor& visitor, const Space&, const Node *node, unsigned axis, const Key& key) {
            if (Base::vol_ == -1) {
                const SO3Root *root = static_cast<const SO3Root*>(node);
                unsigned keyVol = so3::volIndex(key);
                for (unsigned i=0 ; i<4 ; ++i) {
                    if (const Node *c = root->child(Base::vol_ = (keyVol + i) % 4).load(std::memory_order_acquire)) {
                        initVol(key);
                        visitor(c);
                    }
                }
                Base::vol_ = -1;
                distToRegion_ = 0;
            } else {
                const SO3Branch *branch = static_cast<const SO3Branch*>(node);
                Distance dot = so3::dotSplit(branch->split(), key, Base::vol_, axis);
                int childNo = dot > 0;
                recur(visitor, branch, axis, childNo, dot);
                recur(visitor, branch, axis, !childNo, dot);
            }
        }

    };
}


#endif
