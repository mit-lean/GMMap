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
#ifndef NIGH_METRIC_SPACE_SO3_EIGEN_HPP
#define NIGH_METRIC_SPACE_SO3_EIGEN_HPP

#include "space.hpp"
#include <Eigen/Dense>

namespace unc::robotics::nigh::metric {
    template <typename S, int r, int c, int opts, int R, int C>
    struct Space<Eigen::Matrix<S, r, c, opts, R, C>, metric::SO3> : impl::SpaceBase<3> {
        static_assert(
            (r == 1 && (c == Eigen::Dynamic || c == 4)) ||
            (c == 1 && (r == Eigen::Dynamic || r == 4)),
            "SO(3) metric requires a vector with 4 coefficients");

        using Type = Eigen::Matrix<S, r, c, opts, R, C>;
        using Distance = S;
        using Metric = metric::SO3;

        static bool isValid(const Type& a) {
            return std::abs(a.squaredNorm() - 1) <= 4*4 * std::numeric_limits<S>::epsilon();
        }

        static unsigned maxAbsCoeff(const Type& a) {
            unsigned axis;
            a.cwiseAbs().maxCoeff(&axis);
            return axis;
        }

        static Distance coeff(const Type& a, std::size_t index) {
            assert(index < 4);
            return a[index];
        }

        static Distance& coeff(Type& a, std::size_t index) {
            assert(index < 4);
            return a[index];
        }

        Distance distance(const Type& a, const Type& b) const {
            Distance d = std::abs(a.dot(b));
            return d >= 1 ? 0 : std::acos(d);
        }
    };
}

#endif
