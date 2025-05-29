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
#ifndef NIGH_IMPL_SO3_REGION_HPP
#define NIGH_IMPL_SO3_REGION_HPP

#include "region.hpp"
#include "so3.hpp"
#include "constants.hpp"
#include <Eigen/Dense>

namespace unc::robotics::nigh::impl {

    template <typename Key>
    class SO3Region {
        using Distance = typename metric::Space<Key, metric::SO3>::Distance;

        Eigen::Matrix<Distance, 2, 3> min_;
        Eigen::Matrix<Distance, 2, 3> max_;

    public:
        SO3Region() = delete;
        SO3Region(const SO3Region& region)
            : min_(region.min_)
            , max_(region.max_)
        {
        }

        SO3Region(const Key& q, int vol) {
            for (int a=0 ; a<3 ; ++a)
                min_.col(a) = so3::project(q, vol, a);
            max_ = min_;
        }

        void grow(const Key& q, int vol) {
            for (int a=0 ; a<3 ; ++a) {
                Eigen::Matrix<Distance, 2, 1> c = so3::project(q, vol, a);
                assert(-SQRT1_2<Distance> <= c[0] && c[0] <= SQRT1_2<Distance>);
                assert(c[1] >= SQRT1_2<Distance>);
                if (c[0] < min_(0, a))
                    min_.col(a) = c;
                if (c[0] > max_(0, a))
                    max_.col(a) = c;
            }
        }

        Distance selectAxis(unsigned *axis, int vol) const {
            assert(vol != -1);
            Eigen::Matrix<Distance, 1, 3> dots = min_.cwiseProduct(max_).colwise().sum();
            assert((dots.array() >= Distance(0)).all());
            return std::acos(dots.minCoeff(axis));
        }
    };
}

#endif
