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
#ifndef NIGH_METRIC_SPACE_SO2_SCALAR_HPP
#define NIGH_METRIC_SPACE_SO2_SCALAR_HPP

#include "so2.hpp"
#include "../impl/so2.hpp"
#include "space.hpp"
#include "../impl/space_base.hpp"

namespace unc::robotics::nigh::metric {
    template <typename S, int p>
    struct Space<S, SO2<p>, std::enable_if_t<std::is_floating_point_v<S>>>
        : impl::SpaceBase<1>
    {
        using Type = S;
        using Distance = S;
        using Metric = SO2<p>;

        static bool isValid(const Type& a) {
            return std::isfinite(a);
        }

        static Distance coeff(const Type& a, std::size_t i) {
            assert(i == 0);
            return a;
        }

        static Distance& coeff(Type& a, std::size_t i) {
            assert(i == 0);
            return a;
        }

        Distance distance(const Type& a, const Type& b) const {
            return nigh::impl::so2::angularDistance(a, b);
        }
    };
}

#endif
