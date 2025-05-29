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
#ifndef NIGH_IMPL_SO3_HPP
#define NIGH_IMPL_SO3_HPP

#include <cmath>

namespace unc::robotics::nigh::impl::so3 {
    // alias for the distance scalar of an SO(3) state
    template <typename T>
    using dist_t = typename metric::Space<std::decay_t<T>, metric::SO3>::Distance;

    // the split normal for an SO(3) state
    template <typename T>
    using split_t = Eigen::Matrix<dist_t<T>, 2, 1>;

    // coefficient lookup for an SO(3) state
    template <typename T>
    dist_t<T> coeff(const T& t, unsigned i) {
        return metric::Space<std::decay_t<T>, metric::SO3>::coeff(t, i);
    }

    template <typename T>
    static dist_t<T> project1(const T& t, unsigned vol, unsigned axis) {
        assert(vol < 4 && axis < 3);
        split_t<T> r(coeff(t, (vol + axis + 1) % 4), coeff(t, vol));
        dist_t<T> s = (r[1] < 0 ? -r.norm() : r.norm());
        return r[0] / s;
    }

    template <typename T>
    static split_t<T> project(const T& t, unsigned vol, unsigned axis) {
        assert(vol < 4 && axis < 3);
        split_t<T> r(coeff(t, (vol + axis + 1) % 4), coeff(t, vol));
        dist_t<T> s = (r[1] < 0 ? -r.norm() : r.norm());
        r /= s;
        return r;
    }

    template <typename S, typename Q>
    static auto dotSplit(const Eigen::Matrix<S, 2, 1>& split, const Q& q, unsigned vol, unsigned axis) {
        static_assert(std::is_same_v<S, dist_t<Q>>, "split and key coefficient types must match");
        S qv = coeff(q, vol);
        S qa = coeff(q, (vol + axis + 1) % 4);
        S dot = split[1] * qa - split[0] * qv;
        return qv < 0 ? -dot : dot;
    }

    template <typename T>
    static unsigned volIndex(const T& t) {
        using Space = metric::Space<std::decay_t<T>, metric::SO3>;
        return Space::maxAbsCoeff(t);
    }
}

#endif

