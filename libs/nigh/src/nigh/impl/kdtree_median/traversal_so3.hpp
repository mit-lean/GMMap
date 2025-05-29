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
#ifndef NIGH_IMPL_KDTREE_MEDIAN_TRAVERSAL_SO3_HPP
#define NIGH_IMPL_KDTREE_MEDIAN_TRAVERSAL_SO3_HPP

#include "traversal.hpp"
#include "so3_root.hpp"
#include "so3_branch.hpp"
#include "../so3.hpp"
#include <Eigen/Dense>

namespace unc::robotics::nigh::impl::kdtree_median {
    template <class Tree, class Key, class Get>
    class Traversal<Tree, Key, metric::SO3, Get> {
        using Metric = metric::SO3;
        using Space = metric::Space<Key, Metric>;
        using Distance = typename Space::Distance;

        int vol_{-1};
        Distance dotMax_{0};
        Distance distToRegion_{0};
        Eigen::Matrix<Distance, 2, 3> dots_;

        void initVol(const Key& key) {
            Distance qv = Space::coeff(key, vol_);
            Distance s = qv < 0 ? -SQRT1_2<Distance> : SQRT1_2<Distance>;
            dotMax_ = 0;
            for (unsigned a=0 ; a<3 ; ++a) {
                Distance qa = Space::coeff(key, (vol_ + a + 1)%4);
                dots_(0, a) = (qa + qv) * s;
                dots_(1, a) = (qa - qv) * s;
                if (dots_(0, a) < 0 || dots_(1, a) > 0)
                    dotMax_ = std::max(dotMax_, dots_.col(a).cwiseAbs().minCoeff());
            }
            distToRegion_ = std::asin(dotMax_);
        }

        template <class Nearest, class Iter>
        void recur(
            Nearest& nearest, const Node *child, unsigned axis, unsigned splitIndex,
            Distance dot, Iter first, Iter last)
        {
            Distance oldDist = distToRegion_;
            Distance oldDotMax = dotMax_;
            Distance oldDot = dots_(splitIndex, axis);

            dots_(splitIndex, axis) = dot;
            if (dots_(0, axis) < 0 || dots_(1, axis) > 0) {
                Distance minDot = dots_.col(axis).cwiseAbs().minCoeff();
                if (minDot > oldDotMax)
                    distToRegion_ = std::asin(dotMax_ = minDot);
            }

            nearest(child, first, last);

            dots_(splitIndex, axis) = oldDot;
            dotMax_ = oldDotMax;
            distToRegion_ = oldDist;
        }
        
    public:
        Traversal(const Space&) {
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
            if (vol_ < 0) {
                const auto* root = static_cast<const SO3Root*>(node);
                unsigned keyVol = so3::volIndex(key);
                for (unsigned i=0 ; i<4 ; ++i) {
                    vol_ = (keyVol + i) % 4;
                    initVol(key);
                    nearest(root->child(vol_),
                            vol_ == 0 ? first : (first + root->offset(vol_-1)),
                            vol_ == 3 ? last : (first + root->offset(vol_)));
                }
                vol_ = -1;
                distToRegion_ = 0;
            } else {
                const auto* branch = static_cast<const SO3Branch<Distance>*>(node);
                std::array<Iter, 3> iters{{first, first + std::distance(first, last)/2, last}};
                Distance dot = so3::dotSplit(branch->split(), key, vol_, axis);
                int childNo = dot > 0;
                recur(nearest, branch->child(childNo), axis, !childNo, dot,
                      iters[childNo], iters[childNo+1]);
                recur(nearest, branch->child(!childNo), axis, childNo, dot,
                      iters[1-childNo], iters[2-childNo]);
            }
        }
    };
}

#endif
